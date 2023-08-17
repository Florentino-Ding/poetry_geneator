import os
import sys
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm, trange

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class dot_product_attention(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_key: int,
        **kwargs,
    ) -> None:
        super(dot_product_attention, self).__init__(**kwargs)
        self.embed_dim = dim_model
        self.embed2query = nn.Linear(dim_model, dim_key, bias=False)
        self.embed2key = nn.Linear(dim_model, dim_key, bias=False)
        self.embed2value = nn.Linear(dim_model, dim_model, bias=False)
        self.softmax = F.softmax

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        query = self.embed2query(query)
        key = self.embed2key(key)
        value = self.embed2value(value)

        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = self.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value)


class self_attention(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_key: int,
        dot_product_attention=dot_product_attention,
        **kwargs,
    ):
        if not dim_model > 0:
            raise ValueError("embed_dim must be positive")
        if not dim_key > 0:
            raise ValueError("hidden_dim must be positive")
        if not issubclass(dot_product_attention, nn.Module):
            raise ValueError("dot_product_attention must be a subclass of nn.Module")
        super(self_attention, self).__init__(**kwargs)
        self.attention = dot_product_attention(dim_model, dim_key)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        the input should be a tensor with shape (seq_len, batch_size, embed_dim)
        """
        return self.attention(x, x, x, mask)


class multi_head_attention(nn.Module):
    def __init__(
        self,
        dim_model: int,
        key_dim: int,
        head_num: int,
        self_attention=self_attention,
        **kwargs,
    ) -> None:
        super(multi_head_attention, self).__init__(**kwargs)
        self.head_num = head_num
        self.emded2query = nn.ModuleList(
            [nn.Linear(dim_model, key_dim, bias=False) for _ in range(self.head_num)]
        )
        self.emded2key = nn.ModuleList(
            [nn.Linear(dim_model, key_dim, bias=False) for _ in range(self.head_num)]
        )
        self.emded2value = nn.ModuleList(
            [nn.Linear(dim_model, dim_model, bias=False) for _ in range(self.head_num)]
        )
        self.self_attention = self_attention(dim_model, key_dim)

    def forward(self, query, key, value, attention_mask: Optional[torch.Tensor] = None):
        """
        the input should be a tensor with shape (seq_len, batch_size, embed_dim)
        """
        attention_weights = [
            self.self_attention(
                self.emded2query[i](query),
                self.emded2key[i](key),
                self.emded2value[i](value),
                attention_mask,
            )
            for i in range(self.head_num)
        ]
        return torch.cat(attention_weights, dim=-1)


class possitional_encoding(nn.Module):
    def __init__(self, dim_model: int, dropout: int, max_len: int = 1000) -> None:
        super(possitional_encoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        P = torch.zeros((max_len, 1, dim_model))
        position = torch.arange(max_len).unsqueeze(1)
        P[:, 0, 0::2] = torch.sin(
            position.float()
            / torch.pow(10000, 2 * torch.arange(dim_model / 2).float() / dim_model)
        )
        P[:, 0, 1::2] = torch.cos(
            position.float()
            / torch.pow(10000, 2 * torch.arange(dim_model / 2).float() / dim_model)
        )
        self.register_buffer("P", P)

    def forward(self, x):
        """
        the input should be a tensor with shape (seq_len, batch_size, embed_dim)
        """
        x = x + self.P[: x.size(0), :, :].to(x.device)  # type: ignore
        return self.dropout(x)


class transformer(nn.Module):
    def __init__(
        self,
        max_len: int = 125,
        dim_model: int = 512,
        head_num: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dim_key: int = 64,
        P_dropout: float = 0.1,
        activation=F.relu,
        layer_norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super(transformer, self).__init__(**kwargs)
        self.max_len = max_len
        self.dim_model = dim_model
        encoder_layer = transformer_encoder_layer(
            dim_model,
            head_num,
            dim_feedforward,
            dim_key,
            P_dropout,
            activation,
            layer_norm_eps,
        )
        self.encoder = transformer_encoder(encoder_layer, num_encoder_layers)
        decoder_layer = transformer_decoder_layer(
            dim_model,
            head_num,
            dim_feedforward,
            dim_key,
            P_dropout,
            activation,
            layer_norm_eps,
        )
        self.decoder = transformer_decoder(decoder_layer, num_decoder_layers)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        the input should be a tensor with shape (seq_len, batch_size, embed_dim)
        """
        x = self.encoder(x, attention_mask)
        if memory is None:
            memory = x
        x = self.decoder(x, memory, attention_mask, memory_mask)
        return x

    def _init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.training:
            return torch.zeros(
                (self.max_len, batch_size, self.dim_model), device=device
            )
        else:
            return torch.randn(
                (self.max_len, batch_size, self.dim_model), device=device
            )


class transformer_encoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers: int,
        **kwargs,
    ) -> None:
        super(transformer_encoder, self).__init__(**kwargs)
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class transformer_decoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers: int,
        **kwargs,
    ) -> None:
        super(transformer_decoder, self).__init__(**kwargs)
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            target = layer(target, memory, target_mask, memory_mask)
        return target


class transformer_encoder_layer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        head_num: int,
        dim_feedforward: int,
        dim_key: int,
        P_dropout: float,
        activation,
        layer_norm_eps: float,
        **kaargs,
    ) -> None:
        super(transformer_encoder_layer, self).__init__(**kaargs)
        self.self_attention = multi_head_attention(
            dim_model, dim_key, head_num, self_attention
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            activation,
            nn.Linear(dim_feedforward, dim_model),
        )
        self.layer_norm1 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(P_dropout)
        self.dropout2 = nn.Dropout(P_dropout)

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        the input should be a tensor with shape (seq_len, batch_size, embed_dim)
        """
        x = x + self.dropout1(self.self_attention(x, x, x, attention_mask))
        x = self.layer_norm1(x)
        x = x + self.dropout2(self.feed_forward(x))
        x = self.layer_norm2(x)
        return x


class transformer_decoder_layer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        head_num: int,
        dim_feedforward: int,
        dim_key: int,
        P_dropout: float,
        activation,
        layer_norm_eps: float,
        **kaargs,
    ) -> None:
        super(transformer_decoder_layer, self).__init__(**kaargs)
        self.self_attention = multi_head_attention(
            dim_model, dim_key, head_num, self_attention
        )
        self.cross_attention = multi_head_attention(
            dim_model, dim_key, head_num, self_attention
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            activation,
            nn.Linear(dim_feedforward, dim_model),
        )
        self.layer_norm1 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.layer_norm3 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(P_dropout)
        self.dropout2 = nn.Dropout(P_dropout)
        self.dropout3 = nn.Dropout(P_dropout)

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = target + self.dropout1(
            self.self_attention(target, target, target, target_mask)
        )
        target = self.layer_norm1(target)
        target = target + self.dropout2(
            self.cross_attention(target, memory, memory, memory_mask)
        )
        target = self.layer_norm2(target)
        target = target + self.dropout3(self.feed_forward(target))
        target = self.layer_norm3(target)
        return target


class PoetryDataset(Dataset):
    def __init__(self, poet_file, embed_dim: int = 512, max_len: Optional[int] = None):
        super(PoetryDataset, self).__init__()
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
        dataset: PoetryDataset,
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
        dataset: PoetryDataset,
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


def main():
    # 转换为Dataset与train_loader
    poetry_dataset = PoetryDataset(
        poet_file="data/Poetry_data_word2ix_ix2word.npz", max_len=311600
    )
    train_loader = DataLoader(
        poetry_dataset, batch_size=400, shuffle=True, num_workers=8
    )

    # 定义模型
    transformer_model = transformer()
    model = poetry_model(model=transformer_model, epochs=500, device=device)

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


if __name__ == "__main__":
    main()
