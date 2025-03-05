import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import glob
import rasterio
from rasterio.errors import RasterioIOError
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import os
import warnings

# Ignoriamo i warning che non influenzano il funzionamento del codice
warnings.filterwarnings("ignore", category=UserWarning)

# Impostiamo un solo thread per evitare conflitti nei thread interni di PyTorch
torch.set_num_threads(1)

# Normalizzazione delle immagini
def normalize_image(img):
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img.fill(0.5)
    return img

# Dataset personalizzato per caricare i file TIFF
class RadarDataset(Dataset):
    def __init__(self, data_path='../recupero_tiff/cropped/2025/01/', input_length=6, pred_length=6):
        self.input_length = input_length
        self.pred_length = pred_length
        self.seq_length = input_length + pred_length
        self.files = sorted(glob.glob(data_path + '/**/*.tiff', recursive=True))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.valid_files = []
        for file in self.files:
            try:
                with rasterio.open(file) as src:
                    if src.count > 0:
                        self.valid_files.append(file)
            except RasterioIOError:
                print(f"File non valido: {file}")

    def __len__(self):
        return len(self.valid_files) - self.seq_length

    def __getitem__(self, idx):
        images = []
        for i in range(self.seq_length):
            with rasterio.open(self.valid_files[idx + i]) as src:
                img = src.read(1).astype(np.float32)
                img = normalize_image(img)
                img = Image.fromarray(img)
                img = self.transform(img)
                images.append(img)
        images = torch.stack(images)
        # I primi input_length frame sono in input, i successivi pred_length rappresentano la ground truth futura
        input_seq = images[:self.input_length]
        target_seq = images[self.input_length:]
        return input_seq, target_seq

# Definizione della cella ST-LSTM (Spatiotemporal LSTM)
class ST_LSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(ST_LSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv_x = nn.Conv2d(in_channels, 7 * hidden_channels, kernel_size, padding=padding)
        self.conv_m = nn.Conv2d(hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x, h_t, c_t, m_t):
        if h_t is None or c_t is None or m_t is None:
            B, _, H, W = x.size()
            h_t = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
            c_t = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
            m_t = torch.zeros(B, self.hidden_channels, H, W, device=x.device)

        gates_x = self.conv_x(x)
        gates_m = self.conv_m(m_t)

        # Split gates_x into 7 parts
        i_x, f_x, g_x, o_x, i_m, f_m, g_m = torch.split(gates_x, self.hidden_channels, dim=1)

        # Split gates_m into 4 parts, ignoring the last one
        i_xm, f_xm, g_xm, _ = torch.split(gates_m, self.hidden_channels, dim=1)

        # Compute input gate
        i_t = torch.sigmoid(i_x + i_xm)

        # Compute forget gate
        f_t = torch.sigmoid(f_x + f_xm)

        # Compute cell candidate
        g_t = torch.tanh(g_x + g_xm)

        # Update memory cell
        c_new = f_t * c_t + i_t * g_t

        # Compute spatial memory gates
        i_m_t = torch.sigmoid(i_m + self.conv_h(h_t))
        f_m_t = torch.sigmoid(f_m + self.conv_h(h_t))
        g_m_t = torch.tanh(g_m)

        # Update spatial memory
        m_new = f_m_t * m_t + i_m_t * g_m_t

        # Compute output gate
        o_t = torch.sigmoid(o_x + self.conv_h(h_t))

        # Compute hidden state
        h_new = o_t * torch.tanh(self.conv_h(c_new))

        return h_new, c_new, m_new

# Definizione di PredRNN_v2
class PredRNNv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, kernel_size=3):
        super(PredRNNv2, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.cells = nn.ModuleList()

        for i in range(num_layers):
            input_dim = in_channels if i == 0 else hidden_channels
            self.cells.append(ST_LSTMCell(input_dim, hidden_channels, kernel_size))

    def forward(self, x, future_steps=1):
        batch_size, seq_len, _, height, width = x.size()
        h_t, c_t, m_t = [], [], []

        for i in range(self.num_layers):
            zero_state_h = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
            zero_state_c = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
            zero_state_m = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
            h_t.append(zero_state_h)
            c_t.append(zero_state_c)
            m_t.append(zero_state_m)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]
            for i, cell in enumerate(self.cells):
                if i == 0:
                    h_t[i], c_t[i], m_t[i] = cell(x_t, h_t[i], c_t[i], m_t[i])
                else:
                    h_t[i], c_t[i], m_t[i] = cell(h_t[i - 1], h_t[i], c_t[i], m_t[i])

        for t in range(future_steps):
            x_t = h_t[-1]
            for i, cell in enumerate(self.cells):
                if i == 0:
                    h_t[i], c_t[i], m_t[i] = cell(x_t, h_t[i], c_t[i], m_t[i])
                else:
                    h_t[i], c_t[i], m_t[i] = cell(h_t[i - 1], h_t[i], c_t[i], m_t[i])
            outputs.append(h_t[-1])

        outputs = torch.stack(outputs, dim=1)
        return outputs

# Encoder UNet con Skip Connections
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, features=[32, 64]):
        super(UNetEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for feature in features:
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            in_channels = feature

    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)
            # print(f"Skip connection shape: {x.shape}")  # Stampa le dimensioni
        return x, skip_connections

# Decoder UNet con Skip Connections
class UNetDecoder(nn.Module):
    def __init__(self, out_channels=1, features=[64, 32]):
        super(UNetDecoder, self).__init__()
        self.adapt_channels = nn.Conv2d(features[0], features[0], kernel_size=1)
        self.skip_adapters = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=1),  # Adatta il primo skip connection (64 -> 64)
            nn.Conv2d(32, 64, kernel_size=1)   # Adatta il secondo skip connection (32 -> 64)
        ])
        self.channel_adapter = nn.Conv2d(96, 128, kernel_size=1)  # Adapter per uniformare i canali
        self.layers = nn.ModuleList()
        for i in range(len(features) - 1):
            concat_channels = features[i] + features[0]  # Dimensione dopo la concatenazione
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(features[i], features[i + 1], kernel_size=2, stride=2),
                    nn.Conv2d(128, features[i + 1], kernel_size=3, padding=1),  # Ora usa 128 canali
                    nn.ReLU(),
                    nn.Conv2d(features[i + 1], features[i + 1], kernel_size=3, padding=1),
                    nn.ReLU()
                )
            )
        self.final_conv = nn.Conv2d(features[-1], out_channels, kernel_size=1)

    def forward(self, x, skip_connections):
        outputs = []
        for t in range(x.size(1)):
            x_t = x[:, t]
            x_t = self.adapt_channels(x_t)
            for i, layer in enumerate(self.layers):
                x_t = layer[0](x_t)  # Applica ConvTranspose2d
                # print(f"x_t shape after transpose: {x_t.shape}")  # Debug
                skip = skip_connections[i]
                if skip.shape[1] != self.skip_adapters[i].in_channels:
                    raise ValueError(f"Skip connection shape mismatch: expected {self.skip_adapters[i].in_channels} channels, but got {skip.shape[1]}")
                skip = self.skip_adapters[i](skip)
                # print(f"skip shape after adapter: {skip.shape}")  # Debug
                
                # Ridimensiona x_t alle dimensioni di skip
                x_t = F.interpolate(x_t, size=(skip.shape[2], skip.shape[3]), mode='bicubic', align_corners=False)
                
                # Concatena x_t e skip
                x_t = torch.cat([x_t, skip], dim=1)
                # print(f"x_t shape after concat: {x_t.shape}")  # Debug
                
                # Uniforma i canali usando l'adapter
                x_t = self.channel_adapter(x_t)  # Usa l'adapter precedentemente definito
                x_t = nn.ReLU()(x_t)  # Opzionale: aggiungi una ReLU dopo l'adapter
                
                # Applica le operazioni successive
                x_t = layer[1:](x_t)
            x_t = self.final_conv(x_t)
            outputs.append(x_t.unsqueeze(1))
        return torch.cat(outputs, dim=1)

# Blocco residuale EDSR
class EDSRBlock(nn.Module):
    def __init__(self, channels):
        super(EDSRBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.res_scale = 0.1  # Scaling factor per stabilizzare il training

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual * self.res_scale

# Modulo Super Resolution basato su EDSR
class EDSRSuperResolution(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale=2):
        super(EDSRSuperResolution, self).__init__()
        # Primo strato per estrarre le feature iniziali
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # Sequenza di blocchi residuali
        body_blocks = [EDSRBlock(num_features) for _ in range(num_blocks)]
        self.body = nn.Sequential(*body_blocks)
        
        # Strato per espandere i canali in vista della pixel shuffle
        self.tail_conv = nn.Conv2d(num_features, out_channels * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail_conv(x)
        x = self.pixel_shuffle(x)
        return x


# Modello completo RainPredRNN
class RainPredRNN(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, num_layers=2, pred_length=6, use_sr=True):
        super(RainPredRNN, self).__init__()
        self.encoder = UNetEncoder(in_channels=in_channels)
        self.predrnn = PredRNNv2(in_channels=64, hidden_channels=hidden_channels, num_layers=num_layers)
        self.decoder = UNetDecoder(out_channels=1)
        self.pred_length = pred_length
        self.use_sr = use_sr
        if self.use_sr:
            # Utilizza SRCNN come modulo di super resolution
            self.sr_module = EDSRSuperResolution(in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale=2)

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.shape
        encoded = []
        all_skip_connections = []
        for t in range(seq_len):
            encoded_t, skip_connections = self.encoder(x[:, t])
            encoded.append(encoded_t.unsqueeze(1))
            all_skip_connections.append(skip_connections)
        encoded = torch.cat(encoded, dim=1)
        predicted = self.predrnn(encoded, future_steps=self.pred_length)
        last_skip_connections = [sc[-1] for sc in all_skip_connections]
        output = self.decoder(predicted, last_skip_connections)  # output shape: [B, pred_length, C, H, W]
        
        if self.use_sr:
            B, T, C, H, W = output.shape
            # Appiattisce batch e timestep per applicare SRCNN
            output = output.view(B * T, C, H, W)
            output = self.sr_module(output)  # L'upscaling avviene qui
            new_H, new_W = output.shape[2], output.shape[3]
            output = output.view(B, T, C, new_H, new_W)
        return output

# Addestramento del modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = RadarDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)  # Batch size conforme al paper
model = RainPredRNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()

# coefficenti per le metriche di errore, in questo caso c'è equilibri tra dettagli e regolarità
alpha = 1 # 0.7
beta = 0 # 0.3

# Training Loop
epochs = 200
for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # targets ha forma [B, pred_length, C, H, W]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # outputs avrà forma [B, pred_length, C, H, W]
        outputs = model(inputs).float()
        
        # Otteniamo le dimensioni correnti degli output
        B, T, C, H, W = outputs.shape
        
        # Appiattiamo batch e timesteps: diventa [B*T, C, H, W]
        outputs = outputs.view(B * T, C, H, W)
        
        # Ridimensiona gli output alle dimensioni (altezza, larghezza) dei targets
        outputs = F.interpolate(outputs, size=(targets.size(3), targets.size(4)), mode='bicubic', align_corners=False)
        
        # Ripristiniamo la forma originale: [B, T, C, new_H, new_W]
        outputs = outputs.view(B, T, C, targets.size(3), targets.size(4))
        
        targets = targets.float()
        loss = alpha * criterion_mse(outputs, targets) + beta * criterion_l1(outputs, targets)
        if torch.isnan(loss):
            print("Errore: Loss contiene NaN!")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    print(f'Loss: {loss.item():.4f}')


# Visualizzazione e salvataggio della previsione
model.eval()
sample_input, _ = dataset[0]
sample_input = sample_input.unsqueeze(0).to(device)
predicted = model(sample_input).cpu().detach().numpy().squeeze()  # forma: [pred_length, H, W]
predicted = np.clip(predicted, 0, 1)

output_dir = "output_predictions"
os.makedirs(output_dir, exist_ok=True)

for i in range(predicted.shape[0]):
    output_tiff_path = os.path.join(output_dir, f"prediction_frame{i+1}.tiff")
    with rasterio.open(dataset.valid_files[0]) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(output_tiff_path, 'w', **profile) as dst:
            dst.write((predicted[i] * 255).astype(np.uint8), 1)
