import math
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from vertex_voyage.tasks.link_prediction import (
    HadamardLogitsNet,
    IncrementalLogitsDecoder,
    LinkPredictionDataset,
    LinkPredictionSplit,
    QuadraticLogitsNet,
    RandomForestLinkPredictionModelTrainer,
    TorchLogitsLinkPredictionModelTrainer,
    mean_loss_of,
    require_incremental_trainer,
    tensor_dataset_of,
    train_model,
)

VECTOR_DIM = 4
EPOCHS = 2
BATCH_SIZE = 32
PAIR_COUNT_ENDING_IN_BATCH_OF_ONE = BATCH_SIZE + 1
VALIDATION_PAIR_COUNT = 8


def random_pairs(pair_count: int, seed: int = 0):
    generator = np.random.default_rng(seed)
    return (
        generator.normal(size=(pair_count, VECTOR_DIM)),
        generator.normal(size=(pair_count, VECTOR_DIM)),
        generator.integers(0, 2, size=pair_count).astype(float),
    )


def empty_pairs():
    return np.zeros((0, VECTOR_DIM)), np.zeros((0, VECTOR_DIM)), np.zeros(0)


class TestTrainModel(unittest.TestCase):
    def test_last_batch_of_one_pair_trains_for_every_net(self):
        for net_class in (HadamardLogitsNet, QuadraticLogitsNet):
            with self.subTest(net=net_class.__name__):
                net = net_class(vector_dim=VECTOR_DIM)
                _, train_losses, val_losses = train_model(
                    net,
                    *random_pairs(PAIR_COUNT_ENDING_IN_BATCH_OF_ONE),
                    *random_pairs(VALIDATION_PAIR_COUNT, seed=1),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                )
                self.assertEqual(len(train_losses), EPOCHS + 1)
                self.assertTrue(all(math.isfinite(x) for x in train_losses))
                self.assertTrue(all(math.isfinite(x) for x in val_losses))

    def test_empty_validation_split_gives_undefined_validation_loss(self):
        net = HadamardLogitsNet(vector_dim=VECTOR_DIM)
        _, train_losses, val_losses = train_model(
            net,
            *random_pairs(VALIDATION_PAIR_COUNT),
            *empty_pairs(),
            epochs=EPOCHS,
        )
        self.assertEqual(len(val_losses), EPOCHS + 1)
        self.assertTrue(all(math.isnan(x) for x in val_losses))
        self.assertTrue(all(math.isfinite(x) for x in train_losses))

    def test_empty_training_split_is_rejected(self):
        net = HadamardLogitsNet(vector_dim=VECTOR_DIM)
        with self.assertRaises(ValueError):
            train_model(
                net,
                *empty_pairs(),
                *random_pairs(VALIDATION_PAIR_COUNT),
                epochs=EPOCHS,
            )

    def test_training_lowers_the_loss_of_a_separable_split(self):
        torch.manual_seed(0)
        u, v, _ = random_pairs(64)
        y = (np.sum(u * v, axis=1) > 0).astype(float)
        net = HadamardLogitsNet(vector_dim=VECTOR_DIM)
        _, train_losses, _ = train_model(
            net, u, v, y, *empty_pairs(), epochs=30, learning_rate=0.5
        )
        self.assertLess(train_losses[-1], train_losses[0])


class TestMeanLossOf(unittest.TestCase):
    def test_empty_loader_has_undefined_loss(self):
        loader = DataLoader(tensor_dataset_of(*empty_pairs()), batch_size=4)
        net = HadamardLogitsNet(vector_dim=VECTOR_DIM)
        self.assertTrue(math.isnan(mean_loss_of(net, loader)))

    def test_mean_loss_averages_over_batches(self):
        loader = DataLoader(
            tensor_dataset_of(*random_pairs(5)), batch_size=5
        )
        net = HadamardLogitsNet(vector_dim=VECTOR_DIM)
        u_batch, v_batch, y_batch = next(iter(loader))
        expected = net.criterion(
            net(u_batch, v_batch).reshape(-1), y_batch
        ).item()
        self.assertAlmostEqual(mean_loss_of(net, loader), expected, places=6)


DECODER_SEED = 11
REFIT_SEED = 5


def dataset_of(pair_count: int, seed: int = 0) -> LinkPredictionDataset:
    return LinkPredictionDataset(
        train=LinkPredictionSplit(*random_pairs(pair_count, seed)),
        val=LinkPredictionSplit(*empty_pairs()),
    )


def weights_of(decoder: IncrementalLogitsDecoder):
    return [p.detach().clone() for p in decoder.net.parameters()]


def same_weights(first, second) -> bool:
    return all(torch.equal(a, b) for a, b in zip(first, second))


class TestIncrementalLogitsDecoder(unittest.TestCase):
    def decoder(self, learning_rate: float = 0.1, net_class=HadamardLogitsNet):
        trainer = TorchLogitsLinkPredictionModelTrainer(
            net_class, epochs=EPOCHS, learning_rate=learning_rate
        )
        return IncrementalLogitsDecoder(trainer, VECTOR_DIM, DECODER_SEED)

    def test_refit_continues_from_the_current_weights(self):
        decoder = self.decoder(learning_rate=0.0)
        initial_weights = weights_of(decoder)
        decoder.refit(dataset_of(10), REFIT_SEED)
        decoder.refit(dataset_of(10, seed=1), REFIT_SEED)
        self.assertTrue(same_weights(initial_weights, weights_of(decoder)))
        self.assertEqual(decoder.refit_count, 2)

    def test_refit_moves_the_weights(self):
        decoder = self.decoder()
        initial_weights = weights_of(decoder)
        train_losses, val_losses = decoder.refit(dataset_of(10), REFIT_SEED)
        self.assertFalse(same_weights(initial_weights, weights_of(decoder)))
        self.assertEqual(len(train_losses), EPOCHS + 1)
        self.assertTrue(all(math.isnan(x) for x in val_losses))

    def test_global_torch_random_state_is_untouched(self):
        torch.manual_seed(123)
        state_before = torch.random.get_rng_state()
        decoder = self.decoder()
        decoder.refit(dataset_of(40), REFIT_SEED)
        self.assertTrue(torch.equal(state_before, torch.random.get_rng_state()))

    def test_same_seeds_give_the_same_decoder(self):
        weights = []
        for _ in range(2):
            decoder = self.decoder(net_class=QuadraticLogitsNet)
            decoder.refit(dataset_of(40), REFIT_SEED)
            weights.append(weights_of(decoder))
        self.assertTrue(same_weights(*weights))

    def test_probability_is_between_zero_and_one(self):
        decoder = self.decoder()
        probability = decoder.predict_proba(
            np.ones(VECTOR_DIM), np.ones(VECTOR_DIM)
        )
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_invalid_decoders_are_rejected(self):
        hadamard = TorchLogitsLinkPredictionModelTrainer(HadamardLogitsNet)
        invalid = {
            "random forest": lambda: IncrementalLogitsDecoder(
                RandomForestLinkPredictionModelTrainer(), VECTOR_DIM, 0
            ),
            "non-positive dimension": lambda: IncrementalLogitsDecoder(
                hadamard, 0, 0
            ),
            "training vectors of another dimension": lambda: (
                IncrementalLogitsDecoder(hadamard, VECTOR_DIM + 1, 0).refit(
                    dataset_of(4), REFIT_SEED
                )
            ),
        }
        for case, create in invalid.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    create()

    def test_require_incremental_trainer_returns_torch_trainers(self):
        trainer = TorchLogitsLinkPredictionModelTrainer(HadamardLogitsNet)
        self.assertIs(require_incremental_trainer(trainer), trainer)


if __name__ == "__main__":
    unittest.main()
