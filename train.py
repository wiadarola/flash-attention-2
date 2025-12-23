import functools
import logging

import hydra
import omegaconf
import torch
import torchmetrics
from datasets import DatasetDict, load_dataset
from hydra.core.hydra_config import HydraConfig
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer

from model import Transformer


def load_multi30k_dataset() -> DatasetDict:
    """Load, tokenize, and return the Multi30k EN to DE dataset"""

    def tokenize(batch: dict[str, list[str]], tokenizer: PreTrainedTokenizer) -> BatchEncoding:
        return tokenizer(
            batch["de"],
            text_target=batch["en"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    dataset: DatasetDict = load_dataset("bentrevett/multi30k")  # type: ignore
    dataset.set_format(type="torch", columns=["en", "de"])
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    map_fn = functools.partial(tokenize, tokenizer=tokenizer)
    dataset = dataset.map(map_fn, batched=True, remove_columns=["de", "en"])
    return dataset


def prep_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, ...]:
    """Prepare and move inputs, attention masks, and targets from the HuggingFace defaults"""
    x = batch["input_ids"].unsqueeze(-1).to(device, torch.float, non_blocking=True)  # B N C
    y = batch["labels"].to(device, non_blocking=True)  # B N
    mask = torch.where(batch["attention_mask"] == 1, 0, -torch.inf)[:, None, None, :].to(device)
    return x, y, mask


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: omegaconf.DictConfig):
    dataset = load_multi30k_dataset()
    train_loader = DataLoader(dataset["train"], **cfg.dataset.train)
    valid_loader = DataLoader(dataset["validation"], **cfg.dataset.valid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = Transformer(**cfg.model).to(device)
    criterion = nn.CrossEntropyLoss(**cfg.criterion)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer)

    def warmup_inv_sqrt_lr(step_num: int):
        safe_step_num = max(step_num, 1e-9)
        d_model = cfg.model.d_model
        warmup_steps = cfg.trainer.warmup_steps
        return d_model**-0.5 * min(safe_step_num**-0.5, safe_step_num * warmup_steps**-1.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_inv_sqrt_lr)

    mean_loss = torchmetrics.MeanMetric().to(device)
    best_val_loss = torch.inf

    log_dir = HydraConfig.get().run.dir
    logging.info(f"Saving experiment to: {log_dir}")

    for _ in tqdm(range(cfg.trainer.num_epochs), "Epoch"):
        with SummaryWriter(log_dir) as writer:
            # --- Train ---
            model.train()
            mean_loss.reset()
            for batch in tqdm(train_loader, "Train", leave=False):
                x, y, attn_mask = prep_batch(batch, device)

                y_hat = model(x, attn_mask).transpose(1, 2)
                loss: torch.Tensor = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                mean_loss.update(loss)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], scheduler._step_count)
            writer.add_scalar("loss/train", mean_loss.compute(), scheduler._step_count)

            # --- Validate ---
            model.eval()
            mean_loss.reset()
            with torch.no_grad():
                for batch in tqdm(valid_loader, "Valid", leave=False):
                    x, y, attn_mask = prep_batch(batch, device)

                    y_hat = model(x, attn_mask).transpose(1, 2)
                    loss: torch.Tensor = criterion(y_hat, y)

                    mean_loss.update(loss)

            val_loss = mean_loss.compute()
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f"{log_dir}/checkpoints/model_state_best_val.pt")
                best_val_loss = val_loss
            writer.add_scalar("loss/valid", val_loss, scheduler._step_count)

        torch.save(model.state_dict(), f"{log_dir}/checkpoints/model_state_last.pt")


if __name__ == "__main__":
    main()
