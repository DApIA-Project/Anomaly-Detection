import torch
import torch.nn as nn
import torch.nn.functional as F
from _Utils.former_layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from _Utils.former_layers.SelfAttention_Family import FullAttention, AttentionLayer
from _Utils.former_layers.Embed import DataEmbedding
from numpy_typing import np, ax
from _Utils.FeatureGetter import FG_flooding as FG
from torchviz import make_dot


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """


    name = "Transformer"

    def __init__(self, CTX:dict):
        super(Model, self).__init__()
        self.CTX = CTX
        self.pred_len = 1
        self.output_attention = False
        self.task_name = "anomaly_detection"

        # Embedding
        self.enc_embedding = DataEmbedding(CTX["FEATURES_IN"] - CTX["EMBED_IN"],
                                           CTX["D_MODEL"], CTX["EMBED"], CTX["EMBED_IN"], CTX["DROPOUT"])
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, CTX["FACTOR"], attention_dropout=CTX["DROPOUT"],
                                      output_attention=self.output_attention), CTX["D_MODEL"], CTX["N_HEADS"]),
                    CTX["D_MODEL"],
                    CTX["D_FF"],
                    dropout=CTX["DROPOUT"],
                    activation=CTX["ACTIVATION"]
                ) for l in range(CTX["E_LAYERS"])
            ],
            norm_layer=torch.nn.LayerNorm(CTX["D_MODEL"])
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding(CTX["FEATURES_IN"] - CTX["EMBED_IN"], CTX["D_MODEL"], CTX["EMBED"], CTX["EMBED_IN"],
                                               CTX["DROPOUT"])
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, CTX["FACTOR"], attention_dropout=CTX["DROPOUT"],
                                          output_attention=False),
                            CTX["D_MODEL"], CTX["N_HEADS"]),
                        AttentionLayer(
                            FullAttention(False, CTX["FACTOR"], attention_dropout=CTX["DROPOUT"],
                                          output_attention=False),
                            CTX["D_MODEL"], CTX["N_HEADS"]),
                        CTX["D_MODEL"],
                        CTX["D_FF"],
                        dropout=CTX["DROPOUT"],
                        activation=CTX["ACTIVATION"],
                    )
                    for l in range(CTX["D_LAYERS"])
                ],
                norm_layer=torch.nn.LayerNorm(CTX["D_MODEL"]),
                projection=nn.Linear(CTX["D_MODEL"], CTX["FEATURES_OUT"], bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(CTX["D_MODEL"], CTX["FEATURES_OUT"], bias=True)
        if self.task_name == 'anomaly_detection':
            self.flatten = nn.Flatten()
            self.projection = nn.Linear(CTX["D_MODEL"] * CTX["INPUT_LEN"], CTX["FEATURES_OUT"], bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(CTX["DROPOUT"])
            self.projection = nn.Linear(CTX["D_MODEL"] * CTX["seq_len"], CTX["num_class"])

        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=CTX["LEARNING_RATE"])


    def __predict__(self, x):
        # print(x[0])
        gx_enc = FG.get_not(x, "timestamp")
        gx_mark:np.float64_2d[ax.sample, ax.time] = FG.timestamp(x)
        
        x_enc = gx_enc
        x_mark_enc = np.zeros((x_enc.shape[0], x_enc.shape[1], self.CTX["EMBED_IN"]))
        x_mark_enc[:, :, 0] = gx_mark

        x_dec = gx_enc[:, -self.CTX["DEC_LEN"]:]
        x_mark_dec = np.zeros((x_enc.shape[0], self.CTX["DEC_LEN"], 1))
        x_mark_dec[:, :, 0] = gx_mark[:, -self.CTX["DEC_LEN"]:] 

        x_enc = torch.tensor(x_enc, dtype=torch.float32)
        x_mark_enc = torch.tensor(x_mark_enc, dtype=torch.float32)
        x_dec = torch.tensor(x_dec, dtype=torch.float32)
        x_mark_dec = torch.tensor(x_mark_dec, dtype=torch.float32)
        
        y_ = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # clean up
        del x_enc, x_mark_enc, x_dec, x_mark_dec
    

        return y_.reshape(y_.shape[0], self.CTX["FEATURES_OUT"])

    def predict(self, x):
        return self.__predict__(x).detach().numpy()


    def __compute_loss__(self, x, y):
        y = torch.tensor(y, dtype=torch.float32)
        y_ = self.__predict__(x)
        return self.loss(y_, y), y_

    def compute_loss(self, x, y):
        loss, y_ = self.__compute_loss__(x, y)
        return loss.detach().numpy(), y_.detach().numpy()


    def training_step(self, x, y):
        with torch.autograd.set_detect_anomaly(True):
            self.opt.zero_grad()
            loss, output = self.__compute_loss__(x, y)
            loss.backward()
            self.opt.step()
            return loss.detach().numpy(), output.detach().numpy()

    def visualize(self, save_path="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """
        x_enc = torch.randn(1, self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"] - self.CTX["EMBED_IN"])
        x_mark_enc = torch.randn(1, self.CTX["INPUT_LEN"], self.CTX["EMBED_IN"])
        x_dec = torch.randn(1, self.CTX["DEC_LEN"], self.CTX["FEATURES_IN"] - self.CTX["EMBED_IN"])
        x_mark_dec = torch.randn(1, self.CTX["DEC_LEN"], self.CTX["EMBED_IN"])
        
        y = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        make_dot(y.mean(), params=dict(self.named_parameters()), show_attrs=True).render(
            save_path +"/"+ self.name + "_architecture", format="png", cleanup=True)

    def nb_parameters(self):
        """
        Return the number of parameters in the model
        """
        params = self.parameters()
        return sum(p.numel() for p in params if p.requires_grad)

    def get_variables(self):
        """
        Return the variables of the model
        """
        params = self.parameters()
        res = []
        for param in params:
            res.append(param.detach().numpy())
        return res

    def set_variables(self, variables):
        """
        Set the variables of the model
        """
        i = 0
        for param in self.parameters():
            param.data = torch.tensor(variables[i], dtype=torch.float32)
            i += 1


































    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        enc_out = self.flatten(enc_out)  # Flatten the output for anomaly detection
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
