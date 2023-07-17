import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm, trange


class PoetryDataSet(Dataset):
    def __init__(self, poet_file, max_len=None):
        self.max_len = max_len
        self.poet_data = np.load(poet_file, allow_pickle=True)["data"]
        self.word_to_ix = np.load(poet_file, allow_pickle=True)["word2ix"].item()
        self.ix_to_word = np.load(poet_file, allow_pickle=True)["ix2word"].item()

    def __len__(self):
        if not self.max_len:
            return self.poet_data.shape[0]
        else:
            return self.max_len

    def __getitem__(self, idx):
        if self.max_len and idx >= self.max_len:
            raise IndexError("index out of range")
        row = self.poet_data[idx]
        return row

    def vocab_size(self):
        return len(self.word_to_ix)

    def word(self, idx: int):
        return self.ix_to_word[idx]

    def index(self, word: str):
        return self.word_to_ix[word]


class RNN(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_size: int, num_layers: int, **kwargs
    ) -> None:
        super(RNN, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.rnn = nn.RNN(self.vocab_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.vocab_size, num_layers)
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.output_layer = nn.Linear(hidden_size, self.vocab_size)
        else:
            self.num_directions = 2
            self.output_layer = nn.Linear(hidden_size * 2, self.vocab_size)

    def forward(self, input_data, hidden_state):
        input_data = F.one_hot(input_data.T.long(), self.vocab_size).float()
        input_data = input_data.permute(1, 0, 2)  # 将batch_size放在第一维
        input_data, new_hidden_state = self.rnn(input_data, hidden_state)

        return self.output_layer(input_data), new_hidden_state

    def init_state(self, batch_size, device=torch.device("cpu")):
        return torch.randn(
            (
                self.num_directions * self.rnn.num_layers,
                batch_size,
                self.rnn.hidden_size,
            ),
            device=device,
        )


class GRU(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_size: int, num_layers: int, **kwargs
    ) -> None:
        super(GRU, self).__init__()
        self.vocab_size = vocab_size
        self.gru = nn.GRU(self.vocab_size, hidden_size, num_layers)
        self.output_layer = nn.Linear(hidden_size, self.vocab_size)
        if not self.gru.bidirectional:
            self.num_directions = 1
            self.output_layer = nn.Linear(hidden_size, self.vocab_size)
        else:
            self.num_directions = 2
            self.output_layer = nn.Linear(hidden_size * 2, self.vocab_size)

    def forward(self, input_data, hidden_state):
        input_data = F.one_hot(input_data.T.long(), self.vocab_size).float()
        input_data = input_data.permute(1, 0, 2)  # 将batch_size放在第一维
        input_data, new_hidden_state = self.gru(input_data, hidden_state)

        return self.output_layer(input_data), new_hidden_state

    def init_state(self, batch_size, device=torch.device("cpu")):
        return torch.randn(
            (
                self.num_directions * self.gru.num_layers,
                batch_size,
                self.gru.hidden_size,
            ),
            device=device,
        )


class LSTM(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_size: int, num_layers: int, **kwargs
    ) -> None:
        super(LSTM, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(self.vocab_size, hidden_size, num_layers)
        self.output_layer = nn.Linear(hidden_size, self.vocab_size)
        if not self.lstm.bidirectional:
            self.num_directions = 1
            self.output_layer = nn.Linear(hidden_size, self.vocab_size)
        else:
            self.num_directions = 2
            self.output_layer = nn.Linear(hidden_size * 2, self.vocab_size)

    def forward(self, input, hidden_state):
        input = F.one_hot(input.T.long(), self.vocab_size).float()
        input = input.permute(1, 0, 2)  # 将batch_size放在第二维
        input, new_hidden_state = self.lstm(input, hidden_state)

        return self.output_layer(input), new_hidden_state

    def init_state(self, batch_size, device=torch.device("cpu")):
        return (
            torch.randn(
                (
                    self.num_directions * self.lstm.num_layers,
                    batch_size,
                    self.lstm.hidden_size,
                ),
                device=device,
            ),
            torch.randn(
                (
                    self.num_directions * self.lstm.num_layers,
                    batch_size,
                    self.lstm.hidden_size,
                ),
                device=device,
            ),
        )


class peephole_LSTM(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_size: int, num_layers=1, **kwargs
    ) -> None:
        super(peephole_LSTM, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input2forgetgate = nn.Linear(vocab_size, hidden_size)
        self.input2inputgate = nn.Linear(vocab_size, hidden_size)
        self.input2outputgate = nn.Linear(vocab_size, hidden_size)
        self.input2candidate = nn.Linear(vocab_size, hidden_size)
        self.hidden2forgetgate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden2inputgate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden2outputgate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden2candidate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cell2forgetgate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cell2inputgate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cell2outputgate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cell2candidate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_data, hidden_state):
        input_data = F.one_hot(input_data.T.long(), self.vocab_size).float()
        input_data = input_data.permute(1, 0, 2)  # 将batch_size放在第二维
        hidden_state, cell_state = hidden_state
        all_hidden_state = torch.empty(
            (0, input_data.shape[1], self.hidden_size), device=input_data.device
        )
        for time_t_input in input_data:
            self.forgetgate = torch.sigmoid(
                self.input2forgetgate(time_t_input)
                + self.hidden2forgetgate(hidden_state)
                + self.cell2forgetgate(cell_state)
            )
            self.inputgate = torch.sigmoid(
                self.input2inputgate(time_t_input)
                + self.hidden2inputgate(hidden_state)
                + self.cell2inputgate(cell_state)
            )
            self.outputgate = torch.sigmoid(
                self.input2outputgate(time_t_input)
                + self.hidden2outputgate(hidden_state)
                + self.cell2outputgate(cell_state)
            )
            candidate = torch.tanh(
                self.input2candidate(time_t_input)
                + self.hidden2candidate(hidden_state)
                + self.cell2candidate(cell_state)
            )
            cell_state = self.forgetgate * cell_state + self.inputgate * candidate
            hidden_state = self.outputgate * torch.tanh(cell_state)
            all_hidden_state = torch.cat(
                (all_hidden_state, hidden_state.unsqueeze(0)), dim=0
            )

        return self.output_layer(all_hidden_state), (hidden_state, cell_state)

    def init_state(self, batch_size, device=torch.device("cpu")):
        return (
            torch.randn((batch_size, self.hidden_size), device=device),
            torch.randn((batch_size, self.hidden_size), device=device),
        )


class poetry_model:
    def __init__(
        self,
        model,
        lr=1e-3,
        epochs=100,
        loss_func=nn.CrossEntropyLoss,
        optim=torch.optim.Adam,
        clipping_theta=1,
        init_hidden=True,
        log_dir="logs",
        device=torch.device("cpu"),
    ):
        self.device = device
        self.epochs = epochs

        self.model = model.to(self.device)
        self.optimizer = optim(self.model.parameters(), lr=lr)
        self.loss_function = loss_func()
        self.clipping_theta = clipping_theta
        self.init_hidden = init_hidden

        self.writer = SummaryWriter(log_dir=log_dir)

    def _save_data(self, version: int, model_dir=""):
        if model_dir:
            torch.save(
                self.model.state_dict(), "%s/model_%d.pth" % (model_dir, version)
            )
            print("[INFO] Saving to %s_%s.pth" % (model_dir, version))

    def _load_data(self, version, model_dir=""):
        if model_dir:
            self.model.load_state_dict(
                torch.load(
                    "%s/model_%d.pth" % (model_dir, version), map_location=self.device
                )
            )
            print("[INFO] Loading from %s_%s.pth" % (model_dir, version))

    def train(
        self, train_loader, model_dir="model", model_version=0, save_frequency=100
    ):
        if model_version:
            self._load_data(version=model_version, model_dir=model_dir)
            self.epochs -= model_version

        scaler = GradScaler()
        for epoch in trange(self.epochs, desc="Epoch", file=sys.stdout, leave=False):
            loss_list = np.array([])
            hidden_state = None
            t = tqdm(train_loader, desc="Loss", leave=False, file=sys.stdout)
            for vocabulary in t:
                # 准备隐状态（记忆）
                if not hidden_state or self.init_hidden:
                    hidden_state = self.model.init_state(
                        vocabulary.size(0), self.device
                    )
                elif isinstance(hidden_state, tuple):
                    hidden_state = tuple([h.detach_() for h in hidden_state])
                else:
                    hidden_state.detach_()
                # 准备数据
                vocabulary = (
                    vocabulary.long().transpose(1, 0).contiguous().to(self.device)
                )
                self.optimizer.zero_grad()
                input_data, target = vocabulary[:-1, :], vocabulary[1:, :]

                with autocast():
                    output, hidden_state = self.model(input_data, hidden_state)
                    loss = self.loss_function(output.permute(0, 2, 1), target)
                scaler.scale(loss).backward(retain_graph=True)
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clipping_theta
                )
                # output, hidden_state = self.model(input_data, hidden_state)
                # loss = self.loss_function(output.permute(0, 2, 1), target)
                # loss.backward(retain_graph=True)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_theta)

                scaler.step(self.optimizer)
                scaler.update()
                loss_list = np.append(loss_list, np.exp(loss.item()))
                t.set_description(
                    "Epoch %d, Loss: %.4f" % (epoch + 1 + model_version, loss.item())
                )

            # 保存模型参数
            if (epoch + 1 + model_version) % save_frequency == 0 and not epoch == 0:
                self._save_data(version=epoch + 1 + model_version, model_dir=model_dir)
            # 保存loss
            self.writer.add_scalar(
                "Loss", np.mean(loss_list), epoch + 1 + model_version
            )

        if model_version:
            self.epochs += model_version

    def generate_poerty(
        self,
        start_words: str,
        dataset: PoetryDataSet,
        max_gen_len=200,
        model_dir="model",
        model_version=0,
    ) -> str:
        if model_dir:
            assert model_version
            self._load_data(version=model_version, model_dir=model_dir)

        state = self.model.init_state(1, self.device)
        generated_poetry = list()
        # 预热
        for word in start_words:
            input = torch.tensor([[dataset.word_to_ix[word]]], device=self.device)
            _, state = self.model(input, state)
            generated_poetry.append(word)
        # 生成
        while generated_poetry[-1] != "<EOP>" and len(generated_poetry) < max_gen_len:
            output, state = self.model(
                torch.tensor(
                    dataset.word_to_ix[generated_poetry[-1]], device=self.device
                ).reshape(-1, 1),
                state,
            )
            generated_poetry.append(
                dataset.ix_to_word[torch.argmax(output[0, 0]).item()]
            )
        # 格式化
        for i in range(len(generated_poetry)):
            if generated_poetry[i] == "。" or generated_poetry[i] == "！":
                generated_poetry[i] += "\n"

        return "".join(generated_poetry).replace("<EOP>", "")

    def generate_acrostic_poetry(
        self,
        start_words,
        dataset: PoetryDataSet,
        max_gen_len=200,
        model_dir="model",
        model_version=0,
    ):
        if model_dir:
            assert model_version
            self._load_data(version=model_version, model_dir=model_dir)

        state = self.model.init_state(1, self.device)
        start_words_list = list(start_words)
        sentences_list = list()
        for each_word in start_words_list:
            input = torch.tensor([[dataset.word_to_ix[each_word]]], device=self.device)
            _, state = self.model(input, state)
            a_sentence = list()
            a_sentence.append(each_word)
            while (
                not a_sentence[-1] == "。"
                and not a_sentence[-1] == "！"
                and not a_sentence[-1] == "<EOP>"
                and len(a_sentence) < max_gen_len
            ):
                output, state = self.model(
                    torch.tensor(
                        dataset.word_to_ix[a_sentence[-1]], device=self.device
                    ).reshape(-1, 1),
                    state,
                )
                a_sentence.append(dataset.ix_to_word[torch.argmax(output[0, 0]).item()])
            sentences_list.append(a_sentence)
        generated_poetry = str()
        for i in range(len(start_words_list)):
            generated_poetry += "".join(sentences_list[i])
            generated_poetry += "\n"

        return generated_poetry.replace("<EOP>", "")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转换为Dataset与train_loader
    poetry_dataset = PoetryDataSet(
        poet_file="data/Poetry_data_word2ix_ix2word.npz", max_len=311600
    )
    train_loader = DataLoader(
        poetry_dataset, batch_size=400, shuffle=True, num_workers=8
    )

    # 定义模型
    RNN_model = RNN(
        vocab_size=poetry_dataset.vocab_size(), hidden_size=256, num_layers=1
    )
    GRU_model = GRU(
        vocab_size=poetry_dataset.vocab_size(), hidden_size=256, num_layers=2
    )
    LSTM_model = LSTM(
        vocab_size=poetry_dataset.vocab_size(), hidden_size=512, num_layers=1
    )
    peephole_LSTM_model = peephole_LSTM(
        vocab_size=poetry_dataset.vocab_size(), hidden_size=512, num_layers=2
    )
    model = poetry_model(model=LSTM_model, epochs=500, device=device)

    # 模型训练
    model.train(train_loader, model_dir="model", save_frequency=10)

    # 测试生成藏头诗
    start_words = "深度学习"
    acrostic_poetry = model.generate_acrostic_poetry(
        start_words, poetry_dataset, model_dir="model", model_version=100
    )
    print(acrostic_poetry)

    # 测试普通生成诗词
    start_words = "夕阳无限好"
    poetry = model.generate_poerty(
        start_words, poetry_dataset, model_dir="model", model_version=420
    )
    print(poetry)
