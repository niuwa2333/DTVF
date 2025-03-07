import torch.nn as nn
import torch
import h5py
import gradio as gr
import plotly.express as px

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, lstm_out):
        out = self.linear(lstm_out)
        score = torch.bmm(out, out.transpose(1, 2))
        attn = self.softmax(score)
        context = torch.bmm(attn, lstm_out)
        return context

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.attention(out)
        out = out.permute(0, 2, 1)
        out = self.batch_norm(out)
        out = out.permute(0, 2, 1)
        out = self.fc(out[:, -1, :])
        return out

class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop):
        super(CNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.attention = Attention(hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.conv(out)
        out = out.permute(0, 2, 1)
        out = self.attention(out)
        out = out.permute(0, 2, 1)
        out = self.batch_norm(out)
        out = out.permute(0, 2, 1)
        out = self.fc(out[:, -1, :])
        return out

class DualModel(nn.Module):
    def __init__(self, input_size, hidden_size_cnn, hidden_size_lstm, num_layers_cnn, num_layers_lstm, num_classes, drop_cnn, drop_lstm):
        super(DualModel, self).__init__()
        self.cnn = CNNModel(input_size, hidden_size_cnn, num_layers_cnn, num_classes, drop_cnn)
        self.lstm = LSTMModel(input_size, hidden_size_lstm, num_layers_lstm, num_classes, drop_lstm)
        self.weight = nn.Parameter(torch.tensor(0.8))

    def forward(self, x):
        out_cnn = self.cnn(x)
        out_lstm = self.lstm(x)
        out = self.weight * out_cnn + (1 - self.weight) * out_lstm
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('D:\\Dataset_prect\\VF_data\\github\\best_model.pth')
model.eval()
model.to(device)

def load_embeddings_from_h5(file_path):
    with h5py.File(file_path, 'r') as file:
        embeddings = []
        for key in file.keys():
            embedding = file[key][()]
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, 1, -1) 
            elif len(embedding.shape) == 2:
                embedding = embedding.reshape(1, *embedding.shape) 
            elif len(embedding.shape) != 3:
                raise ValueError("The embedding must be a 1D, 2D, or 3D tensor.")
            embeddings.append(embedding)
    return embeddings

def predict_vf(file_obj):
    embeddings = load_embeddings_from_h5(file_obj.name)


    input_tensors = [torch.tensor(embedding, dtype=torch.float32).to(device) for embedding in embeddings]

    probabilities = []
    with torch.no_grad():
        for input_tensor in input_tensors:
            output = model(input_tensor)
            probability = torch.sigmoid(output)[0][1].item()  
            probabilities.append(probability)

    positive_count = sum(1 for prob in probabilities if prob >= 0.5)
    negative_count = len(probabilities) - positive_count
    total_count = len(probabilities)
    positive_ratio = positive_count / total_count
    negative_ratio = negative_count / total_count

    labels = ['Positive', 'Negative']
    values = [positive_ratio, negative_ratio]
    fig = px.pie(names=labels, values=values, title='Positive vs Negative Predictions')

    return probabilities, fig

iface = gr.Interface(
    fn=predict_vf,  
    inputs=gr.components.File(label="Upload .h5 file containing the embeddings"),
    outputs=[gr.components.Textbox(label="Predicted Probabilities"), gr.components.Plot(label="Pie Chart")], 
    title="VF Prediction Model",
    description="Upload an .h5 file containing the embeddings to get the probabilities of them being VF (Virulence Factor) and a pie chart showing the distribution of positive and negative predictions."
)
iface.launch(share=False, inbrowser=True)
