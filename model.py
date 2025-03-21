import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientChannelAttention(nn.Module):
    """Efficient Channel Attention module for focusing on important features"""
    
    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        t = int(abs((torch.log2(torch.tensor(channels)) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        
    def forward(self, x):
        # x: [B, C, T]
        y = self.avg_pool(x)  # [B, C, 1]
        y = y.transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = y.transpose(-1, -2)  # [B, C, 1]
        return x * y.expand_as(x)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class HandwritingGenerationModel(nn.Module):
    """Model for handwriting generation combining LSTM and Transformer attention"""
    
    def __init__(self, input_size=3, hidden_size=512, num_layers=3, 
                 vocab_size=80, embedding_dim=256, dropout=0.3):
        super(HandwritingGenerationModel, self).__init__()
        
        # Text embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Text encoder (bidirectional LSTM)
        self.text_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Feature extractor for handwriting
        self.stroke_encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_size),
            nn.ReLU()
        )
        
        # Channel attention
        self.channel_attention = EfficientChannelAttention(hidden_size)
        
        # LSTM decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size + hidden_size*2,  # stroke features + text context
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Transformer layers for global attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)
        
        # Output layers
        self.stroke_predictor = nn.Linear(hidden_size, input_size)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, 8, dropout=dropout)
        
    def _encode_text(self, text, text_lengths):
        """Encode the input text"""
        # Embed text
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        
        # Pack padded sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Pass through LSTM
        outputs, (hidden, cell) = self.text_encoder(packed)
        
        # Unpack sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, (hidden, cell)
    
    def _get_attention_context(self, decoder_state, text_encodings, text_lengths):
        """Get context vector using attention mechanism"""
        # Prepare decoder state for attention
        query = decoder_state.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Create attention mask based on text lengths
        batch_size, max_len = text_encodings.size(0), text_encodings.size(1)
        mask = torch.arange(max_len, device=text_lengths.device).expand(batch_size, max_len) >= text_lengths.unsqueeze(1)
        
        # Apply attention
        context, _ = self.attention(
            query=query.transpose(0, 1),
            key=text_encodings.transpose(0, 1),
            value=text_encodings.transpose(0, 1),
            key_padding_mask=mask
        )
        
        return context.transpose(0, 1).squeeze(1)
    
    def forward(self, stroke, text, stroke_lengths, text_lengths, teacher_forcing_ratio=1.0):
        """Forward pass"""
        batch_size, seq_len = stroke.size(0), stroke.size(1)
        
        # Encode text
        text_encodings, (hidden, cell) = self._encode_text(text, text_lengths)
        
        # Initialize decoder input (start with zeros)
        decoder_input = torch.zeros(batch_size, 1, stroke.size(2), device=stroke.device)
        
        # Initialize decoder hidden state and cell with encoded text
        decoder_hidden = hidden
        decoder_cell = cell
        
        # Initialize output container
        outputs = torch.zeros(batch_size, seq_len, stroke.size(2), device=stroke.device)
        
        # Decode sequence step by step
        for t in range(seq_len):
            # Encode current stroke point
            stroke_features = self.stroke_encoder(decoder_input.squeeze(1))
            
            # Get attention context from text encoding
            context = self._get_attention_context(stroke_features, text_encodings, text_lengths)
            
            # Combine stroke features with context
            combined = torch.cat([stroke_features, context], dim=1).unsqueeze(1)
            
            # Pass through LSTM decoder
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                combined, (decoder_hidden, decoder_cell)
            )
            
            # Apply transformer for global attention
            decoder_output = decoder_output.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
            decoder_output = self.positional_encoding(decoder_output)
            transformer_output = self.transformer_encoder(decoder_output)
            transformer_output = transformer_output.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
            
            # Apply channel attention
            transformer_output = transformer_output.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
            attended_output = self.channel_attention(transformer_output)
            attended_output = attended_output.transpose(1, 2)  # [batch_size, seq_len, hidden_size]
            
            # Predict next stroke point
            prediction = self.stroke_predictor(attended_output.squeeze(1))
            outputs[:, t] = prediction
            
            # Teacher forcing: use real target data as next input with probability
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            next_input = stroke[:, t].unsqueeze(1) if teacher_force and t < seq_len - 1 else prediction.unsqueeze(1)
            decoder_input = next_input
        
        return outputs

class HandwritingGenerator(nn.Module):
    """Generator model for producing handwritten text"""
    
    def __init__(self, model_path, vocab_size, device='cuda'):
        super(HandwritingGenerator, self).__init__()
        self.model = HandwritingGenerationModel(vocab_size=vocab_size)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device
        self.model.to(device)
        
    def generate(self, text, char_to_idx, mean, std, max_length=1000):
        """Generate handwriting for given text"""
        # Convert text to indices
        text_indices = [char_to_idx.get(char, 0) for char in text]
        text_tensor = torch.tensor(text_indices, dtype=torch.long).unsqueeze(0).to(self.device)
        text_length = torch.tensor([len(text_indices)], dtype=torch.long).to(self.device)
        
        # Initialize stroke
        stroke = torch.zeros(1, 1, 3, device=self.device)
        
        # Generate strokes point by point
        generated_strokes = []
        hidden = None
        cell = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                output, (hidden, cell) = self.model(
                    stroke=stroke,
                    text=text_tensor,
                    stroke_lengths=torch.tensor([1], device=self.device),
                    text_lengths=text_length,
                    teacher_forcing_ratio=0.0
                )
                
                # Sample from output distribution
                stroke = output
                
                # Append to generated strokes
                generated_strokes.append(stroke.cpu().numpy()[0])
                
                # Check for end of sequence (pen up)
                if stroke[0, 0, 0] > 0.9:  # High probability of pen up
                    break
        
        # Concatenate all strokes
        generated_strokes = np.concatenate(generated_strokes, axis=0)
        
        # Denormalize
        generated_strokes = generated_strokes * std + mean
        
        return generated_strokes
