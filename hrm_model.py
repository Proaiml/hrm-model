import torch
import torch.nn as nn
import torch.nn.functional as F



class SwiGLU(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size,hidden_size*4)
        self.layer2 = nn.Linear(hidden_size,hidden_size*4)
        self.layer3 = nn.Linear(hidden_size*4,hidden_size)

    def forward(self,x):
        a = self.layer1(x)
        b = self.layer2(x)

        gated = F.silu(a) * b
        output = self.layer3(gated)

        return output



class HRMBlock(nn.Module):
    def __init__(self,hidden_size,num_heads):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first = True

        )
        self.norm1 = nn.RMSNorm(hidden_size)

        self.mlp = SwiGLU(hidden_size)

        self.norm2 = nn.RMSNorm(hidden_size)


    def forward(self,x):
        attention_out, _ = self.attention(x,x,x)
        x=self.norm1(x+attention_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)

        return x


class ReasoningModule(nn.Module):
    def __init__(self,hidden_size,num_heads,num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            HRMBlock(hidden_size,num_heads)
            for _ in range(num_layers)
        ])

    def forward(self,hidden_state,input_injection):
        x = hidden_state + input_injection
        for layer in self.layers:
            x = layer(x)

        return x

class TinyHrm(nn.Module):
    def __init__(self,hidden_size,num_heads,H_layers,L_layers,H_cycles,L_cycles):
        super().__init__()
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.L_level= ReasoningModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=L_layers
        )
        self.H_level= ReasoningModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=H_layers
        )

    def forward(self,x):
        z_H = torch.zeros_like(x)
        z_L = torch.zeros_like(x)

        for  _ in range(self.H_cycles):
            for _ in range(self.L_cycles):
                z_L=self.L_level(z_L,z_H+x)

            z_H=self.H_level(z_H, z_L )

        return z_H,z_L



if __name__ == "__main__":
    model = TinyHrm(hidden_size=128,num_heads=4,H_layers=1,L_layers=1,H_cycles=2,L_cycles=3)
    x = torch.randn(1,64,128)
    z_h, z_l  = model(x)




    print("giriş",x.shape)
    print("z-l",z_l.shape)
    print("z_h",z_h.shape)
    import matplotlib.pyplot as plt

    # -------------------------------------------------
    # BASİT 8x8 ÖRNEK
    # A = 2
    # B = 3
    # boş = 0
    # engel = 1
    # -------------------------------------------------

    grid = torch.tensor([[
        2, 0, 0, 0, 0, 0, 0, 3,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    ]], dtype=torch.long)

    # Her hücreyi 128 boyuta çıkar
    embedding = nn.Embedding(4, 128)

    # 0=YUKARI, 1=AŞAĞI, 2=SOL, 3=SAĞ
    head = nn.Linear(128, 4)

    model = TinyHrm(
        hidden_size=128,
        num_heads=4,
        H_layers=1,
        L_layers=1,
        H_cycles=2,
        L_cycles=3
    )

    optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(embedding.parameters()) +
        list(head.parameters()),
        lr=0.001
    )

    loss_fn = nn.CrossEntropyLoss()

    # Doğru cevap SAĞ
    target = torch.tensor([3])

    losses = []

    for epoch in range(100):

        x = embedding(grid)

        z_h, z_l = model(x)

        # A ilk hücrede olduğu için ilk tokenı alıyoruz
        start_state = z_h[:, 0, :]

        logits = head(start_state)

        loss = loss_fn(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 10 == 0:
            prediction = logits.argmax(dim=1).item()

            print(
                "Epoch:", epoch,
                "Loss:", round(loss.item(), 4),
                "Tahmin:", prediction
            )

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("HRM Basit A → B Testi")
    plt.show()