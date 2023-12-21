import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.utils.data.dataset import Dataset, random_split
from tqdm import tqdm

from ppo import ActorCritic
from swap_env import SwapEnv
from utils import get_config, get_cost, to_device


class BC(LightningModule):
    def __init__(
        self,
        lr: float = 0.01,
        lr_gamma: float = 0.99,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.actor_critic = ActorCritic(**kwargs["model_params"])
        self.cel = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        data, target = batch
        for state in data:
            to_device(state, self.device)
        pis, actions = self.actor_critic.actor_forward(data)
        logits = pis.logits.transpose(-1, -2)
        loss = self.cel(logits, target.long())
        self.log("loss/train_loss", loss, prog_bar=True, logger=True)
        self.log(
            "loss/train_acc",
            (logits.argmax(-2) == target.long()).float().mean(),
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        for state in data:
            to_device(state, self.device)
        pis, actions = self.actor_critic.actor_forward(data)
        logits = pis.logits.transpose(-1, -2)
        loss = self.cel(logits, target.long())
        self.log("loss/val_loss", loss, prog_bar=True, logger=True)
        self.log(
            "loss/val_acc",
            (logits.argmax(-2) == target.long()).float().mean(),
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.lr_gamma
        )
        return [optimizer], [scheduler]


class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)


def collect_data(env_params, num_interactions):
    def greedy_act(state):
        city_pop = state["population"]
        facility_list = state["facility_list"]
        distance_m = state["distance_m"]
        mask = state["mask"]
        fac_in_indices = np.where(mask == 1)[0]

        min_cost = np.inf

        for i, fac_out in enumerate(facility_list):
            for fac_in in fac_in_indices:
                facility_list_ = facility_list.copy()
                facility_list_[i] = fac_in
                cost = get_cost(facility_list_, distance_m, city_pop)
                if cost < min_cost and cost < state["total_cost"]:
                    min_cost = cost
                    best_action = (fac_out, fac_in)
                del facility_list_

        if min_cost == np.inf:
            best_action = (0, 0)
        return best_action

    env = SwapEnv(**env_params)

    state, info = env.reset()

    expert_obs = []
    expert_actions = []

    for i in tqdm(range(num_interactions)):
        action = greedy_act(state)
        if action[0] != action[1]:
            expert_obs.append(state)
            expert_actions.append(action)
        else:
            i -= 1
        state, reward, done, truncated, info = env.step(action)
        if done or truncated:
            state, info = env.reset()

    expert_obs = np.array(expert_obs)
    expert_actions = np.array(expert_actions)
    np.savez_compressed("expert_graph.npz", obs=expert_obs, actions=expert_actions)


def collate_fn_bc(batch):
    states, actions = zip(*batch)
    actions = torch.as_tensor(actions, dtype=torch.long)
    return states, actions


def train_BC(batch_size):
    expert_obs, expert_actions = (
        np.load("expert_graph.npz", allow_pickle=True)["obs"],
        np.load("expert_graph.npz", allow_pickle=True)["actions"],
    )
    expert_dataset = ExpertDataSet(expert_obs, expert_actions)

    train_size = int(0.8 * len(expert_dataset))
    test_size = len(expert_dataset) - train_size
    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_expert_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_bc,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_expert_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_bc,
    )

    tb_logger = pl_loggers.TensorBoardLogger("./logs/bc/", name="")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="loss/val_acc", mode="max", save_last=True
    )
    trainer = Trainer(
        callbacks=[LearningRateMonitor(), checkpoint_callback],
        logger=tb_logger,
        **config["bc_trainer"],
    )
    model = BC(**config["bc_model"])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    config = get_config(["-c", "config/train.yaml"])
    collect_data(config["bc"]["env_params"], config["bc"]["num_interactions"])
    train_BC(config["bc"]["batch_size"])
