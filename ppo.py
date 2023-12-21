import os
import time
from typing import Callable, Iterator, List, Tuple

import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
import yaml
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.distributions import Categorical
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, IterableDataset

from swap_env import SwapEnv
from utils import get_config, to_device


def collate_fn_ppo(batch):
    states, actions, logp_olds, v_olds, qvals, advs = zip(*batch)
    actions = torch.stack(actions)
    logp_olds = torch.stack(logp_olds)
    v_olds = torch.as_tensor(v_olds, dtype=torch.float).unsqueeze(-1)
    qvals = torch.as_tensor(qvals, dtype=torch.float).unsqueeze(-1)
    advs = torch.as_tensor(advs, dtype=torch.float).unsqueeze(-1)
    return (states, actions, logp_olds, v_olds, qvals, advs)


class GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        num_layers=2,
        layer_name="GCN",
        **kwargs,
    ):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()

        gnn_layer = getattr(geom_nn, layer_name)

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
            ]
            in_channels = c_hidden * kwargs["heads"] if "heads" in kwargs else c_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                x = layer(x)

        return x


class MLPModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of hidden layers
        """
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU(inplace=True)]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


class ActorCritic(nn.Module):
    def __init__(
        self, fac_c_in, c_hidden, c_out, num_layers, layer_name, **kwargs
    ) -> None:
        super().__init__()
        if "heads" not in kwargs:
            kwargs["heads"] = 1
        self.fac_gnn = GNNModel(
            fac_c_in, c_hidden, c_out, num_layers, layer_name, **kwargs
        )
        self.actor_prob = MLPModel(c_out * kwargs["heads"], c_hidden, 1, num_layers)
        self.att = nn.Linear(
            c_out * kwargs["heads"], c_out * kwargs["heads"], bias=False
        )

        self.critic_gnn = GNNModel(
            fac_c_in, c_hidden, c_out, num_layers, layer_name, **kwargs
        )
        self.critic = MLPModel(c_out * kwargs["heads"], c_hidden, 1, num_layers)

    def forward(self, state):
        if type(state) is not dict:
            logits, actions = [], []
            for bs in range(len(state)):
                logit, _ = self.actor_forward_single(state[bs])
                logits.append(logit)
                action = torch.argmax(logit, dim=-1)
                actions.append(action)

            return torch.stack(actions)
        else:
            logit, _ = self.actor_forward_single(state)
            action = torch.argmax(logit, dim=-1)
            return action

    def actor_forward(self, state):
        if type(state) is not dict:
            logits, actions = [], []
            for bs in range(len(state)):
                logit, action = self.actor_forward_single(state[bs])
                logits.append(logit)
                actions.append(action)

            return Categorical(logits=torch.stack(logits)), torch.stack(actions)

        else:
            logit, action = self.actor_forward_single(state)
            return Categorical(logits=logit), action

    def actor_forward_single(self, state):
        batch_fac, mask = state["fac_data"], state["mask"]
        emb_fac = self.fac_gnn(batch_fac.x, batch_fac.edge_index, batch_fac.edge_attr)
        act_scores1 = self.actor_prob(emb_fac).squeeze()
        mask1 = torch.where(mask, -float("inf"), 0)  # choose fac_out
        logits1 = act_scores1 + mask1

        pi1 = Categorical(logits=logits1)
        action1 = pi1.sample()

        feat_act = torch.tanh(self.att(emb_fac[action1]))
        act_scores2 = torch.matmul(emb_fac, feat_act.unsqueeze(1)).sum(dim=-1)
        mask2 = torch.where(mask, 0, -float("inf"))  # choose fac_in
        logits2 = act_scores2 + mask2
        pi2 = Categorical(logits=logits2)
        action2 = pi2.sample()

        logits = torch.stack([logits1, logits2])
        action = torch.stack([action1, action2])

        return logits, action

    def critic_forward(self, state):
        if type(state) is not dict:
            scores = []
            for bs in range(len(state)):
                batch_fac = state[bs]["fac_data"]
                emb_fac = self.critic_gnn(
                    batch_fac.x, batch_fac.edge_index, batch_fac.edge_attr
                )
                emb_global = geom_nn.global_mean_pool(emb_fac, None)
                scores.append(self.critic(emb_global))
            return torch.stack(scores)
        else:
            batch_fac = state["fac_data"]

            emb_fac = self.critic_gnn(
                batch_fac.x, batch_fac.edge_index, batch_fac.edge_attr
            )
            emb_global = geom_nn.global_mean_pool(emb_fac, None)
            score = self.critic(emb_global)
            return score

    def get_log_prob(self, pi: Categorical, actions: torch.Tensor):
        return pi.log_prob(actions)

    def actor_loss(
        self, state, action, logp_old, qval, adv, clip_ratio
    ) -> torch.Tensor:
        pi, _ = self.actor_forward(state)
        logp = self.get_log_prob(pi, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()

        with torch.no_grad():
            log_ratio = logp - logp_old
            approx_kl_div = (
                torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy().item()
            )

        clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_ratio).float()).item()

        entropy_loss = -pi.entropy().mean()
        return loss_actor, entropy_loss, approx_kl_div, clip_fraction

    def critic_loss(
        self, state, action, logp_old, v_old, qval, adv, clip_ratio
    ) -> torch.Tensor:
        value = self.critic_forward(state)
        clip_value = v_old + torch.clamp(value - v_old, -clip_ratio, clip_ratio)
        v_max = torch.max((qval - value).pow(2), (qval - clip_value).pow(2))
        loss_critic = v_max.mean()
        return loss_critic


class ExperienceSourceDataset(IterableDataset):
    """Implementation from PyTorch Lightning Bolts: https://github.com/Lightning-AI/lightning-
    bolts/blob/master/pl_bolts/datamodules/experience_source.py.

    Basic experience source dataset. Takes a generate_batch function that returns an iterator. The logic for the
    experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        iterator = self.generate_batch()
        return iterator


class PPOLightning(LightningModule):
    def __init__(
        self,
        env_params: dict = None,
        model_params: dict = None,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 5e-4,
        lr_gamma: float = 0.99,
        batch_size: int = 512,
        steps_per_epoch: int = 2048,
        nb_optim_iters: int = 4,
        clip_ratio: float = 0.2,
        clip_decay: float = 1,
        ent_weight: float = 0.01,
        critic_weight: float = 0.5,
        gradient_clip_val: float = None,
        mode: str = "train",
        **kwargs,
    ) -> None:
        """
        Args:
            env_params: env parameters
            model_params: model parameters
            gamma: discount factor
            lam: advantage discount factor (lambda in the paper)
            lr: learning rate
            lr_gamma: learning rate decay rate
            batch_size:  batch_size when training network- can simulate number of policy updates performed per epoch
            steps_per_epoch: how many action-state pairs to rollout for trajectory collection per epoch
            nb_optim_iters: how many steps of gradient descent to perform on each batch
            clip_ratio: hyperparameter for clipping in the policy objective
            clip_decay: clip_ratio decay rate
            ent_weight: weight of the entropy loss term
            critic_weight: weight of the critic loss term
            gradient_clip_val: gradient clipping
            mode: train or test
        """
        super().__init__()

        self.save_hyperparameters()

        if mode == "test":
            self.actor_critic = ActorCritic(**self.hparams.model_params)
            return

        self.automatic_optimization = False

        self.env = SwapEnv(**env_params)

        self.actor_critic = ActorCritic(**self.hparams.model_params)

        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []
        self.batch_v = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

        self.state = self.env.reset()[0]

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi, action = self.actor_critic.actor_forward(x)
        value = self.actor_critic.critic_forward(x)

        return pi, action, value

    def predict(self, x):
        return self.actor_critic(x)

    def discount_rewards(self, rewards: List[float], discount: float) -> List[float]:
        """Calculate the discounted rewards of all rewards in list.

        Args:
            rewards: list of rewards/advantages

        Returns:
            list of discounted rewards/advantages
        """
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(
        self, rewards: List[float], values: List[float], last_value: float
    ) -> List[float]:
        """Calculate the advantage given rewards, state values, and the last value of episode.

        Args:
            rewards: list of episode rewards
            values: list of state values from critic
            last_value: value of last state of episode

        Returns:
            list of advantages
        """
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [
            rews[i] + self.hparams.gamma * vals[i + 1] - vals[i]
            for i in range(len(rews) - 1)
        ]
        adv = self.discount_rewards(delta, self.hparams.gamma * self.hparams.lam)

        return adv

    def generate_trajectory_samples(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Contains the logic for generating trajectory data to train policy and value network
        Yield:
           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage
        """

        for step in range(self.hparams.steps_per_epoch):
            to_device(self.state, self.device)

            with torch.no_grad():
                pi, action, value = self(self.state)
                log_prob = self.actor_critic.get_log_prob(pi, action)
                self.logger.log_metrics(
                    {"entropy/pi": pi.entropy().mean()}, self.global_step
                )

            next_state, reward, done, truncated, _ = self.env.step(action.cpu().numpy())

            self.episode_step += 1

            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)
            self.batch_v.append(value)

            self.ep_rewards.append(reward)
            self.ep_values.append(value.item())

            self.state = next_state

            epoch_end = step == (self.hparams.steps_per_epoch - 1)

            if epoch_end or done or truncated:
                # if trajectory ends abtruptly, bootstrap value of next state
                if not done:
                    to_device(self.state, self.device)
                    with torch.no_grad():
                        _, _, value = self(self.state)
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                # discounted cumulative reward
                self.batch_qvals += self.discount_rewards(
                    self.ep_rewards + [last_value], self.hparams.gamma
                )[:-1]
                # advantage
                self.batch_adv += self.calc_advantage(
                    self.ep_rewards, self.ep_values, last_value
                )
                # logs
                self.epoch_rewards.append(sum(self.ep_rewards))
                # reset params
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0
                self.state = self.env.reset()[0]

            if epoch_end:
                train_data = zip(
                    self.batch_states,
                    self.batch_actions,
                    self.batch_logp,
                    self.batch_v,
                    self.batch_qvals,
                    self.batch_adv,
                )
                for state, action, logp_old, v_old, qval, adv in train_data:
                    yield state, action, logp_old, v_old, qval, adv

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_v.clear()
                self.batch_qvals.clear()

                # logging
                self.avg_reward = sum(self.epoch_rewards) / self.hparams.steps_per_epoch

                # if epoch ended abruptly, exlude last cut-short episode to prevent stats skewness
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (
                    self.hparams.steps_per_epoch - steps_before_cutoff
                ) / nb_episodes

                self.epoch_rewards.clear()

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/avg_ep_reward": -1})

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        """Carries out a single update to actor and critic network from a batch of replay buffer.

        Args:
            batch: batch of replay buffer/trajectory data
        """
        state, action, old_logp, v_old, qval, adv = batch

        if self.hparams.batch_size > 1:
            adv = (adv - adv.mean()) / adv.std()

        self.log(
            "hp/avg_ep_len",
            self.avg_ep_len,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "hp/avg_ep_reward",
            self.avg_ep_reward,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "hp/avg_reward",
            self.avg_reward,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        (
            loss_actor,
            entropy_loss,
            approx_kl_div,
            clip_fraction,
        ) = self.actor_critic.actor_loss(
            state, action, old_logp, qval, adv, self.hparams.clip_ratio
        )

        loss_critic = self.actor_critic.critic_loss(
            state, action, old_logp, v_old, qval, adv, self.hparams.clip_ratio
        )
        loss = (
            self.hparams.ent_weight * entropy_loss
            + loss_actor
            + self.hparams.critic_weight * loss_critic
        )

        self.manual_backward(loss)
        if self.hparams.gradient_clip_val is not None:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.hparams.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx + 1 == self.hparams.steps_per_epoch // self.hparams.batch_size:
            scheduler.step()
            self.hparams.clip_ratio *= self.hparams.clip_decay

        self.log(
            "loss/loss_critic",
            loss_critic,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "loss/loss_actor",
            loss_actor,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "loss/loss_entropy",
            entropy_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "loss/clip_ratio",
            self.hparams.clip_ratio,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "loss/approx_kl_div",
            approx_kl_div,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "loss/clip_fraction",
            clip_fraction,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.lr_gamma
        )
        return [optimizer], [scheduler]

    def optimizer_step(self, *args, **kwargs):
        """Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic for each data
        sample."""
        for _ in range(self.hparams.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = ExperienceSourceDataset(self.generate_trajectory_samples)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_fn_ppo,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()


def train_ppo(config):
    seed = config["logging"]["seed"]
    if type(seed) is int:
        seed_everything(seed)
        config["ppo_trainer"]["deterministic"] = True
        config["ppo_trainer"]["benchmark"] = False

    # logging
    log_path = "./logs/ppo/"
    log_name = eval(config["logging"]["log_name"])
    os.makedirs(f"{log_path}/{log_name}", exist_ok=True)
    yaml.safe_dump(config, open(f"{log_path}/{log_name}/config.yaml", "w"))
    print("logging saved at", f"{log_path}/{log_name}/config.yaml")

    # trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_path, name="", version=log_name, default_hp_metric=False
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="hp/avg_ep_reward", mode="max", save_last=True
    )
    trainer = Trainer(
        callbacks=[LearningRateMonitor(), checkpoint_callback],
        logger=tb_logger,
        **config["ppo_trainer"],
    )
    if config["ckpt"] is not None:
        model = PPOLightning.load_from_checkpoint(config["ckpt"], **config["ppo"])
    else:
        model = PPOLightning(**config["ppo"])
    trainer.fit(model)


if __name__ == "__main__":
    config = get_config(["-c", "config/train.yaml"])
    train_ppo(config)
