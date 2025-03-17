from .simulator import Event, State, InvalidEventError
from .model import Gene, Synteny, Species
from numpy.random import default_rng
from unittest import TestCase


class TestState(TestCase):
    def setUp(self):
        Gene.__dataclass_fields__["id"].default_factory.reset()
        Synteny.__dataclass_fields__["id"].default_factory.reset()
        Species.__dataclass_fields__["id"].default_factory.reset()

        self.generator = default_rng(42)
        self.state = State.unit()
        self.state.generator = self.generator

    def test_serialize(self):
        self.assertEqual(str(self.state), "S1: {X1: (G1)}")
        self.assertEqual(
            self.state.serialize(),
            {"S1": self.state["S1"].serialize()}
        )
        self.assertEqual(
            State.unserialize(self.state.serialize()).species,
            self.state.species,
        )

    def test_speciation(self):
        self.state.speciation()
        self.assertEqual(str(self.state), "S2: {X2: (G2)}, S3: {X3: (G3)}")

        self.state.speciation("S3")
        self.assertEqual(
            str(self.state),
            "S2: {X2: (G2)}, S4: {X4: (G4)}, S5: {X5: (G5)}"
        )

        g6, g7, g8, g9 = Gene(), Gene(), Gene(), Gene()
        self.state["S2"]["X2"][2:] = [g6, g7]
        self.state["S5"]["X5"][2:] = [g8, g9]
        self.assertEqual(
            str(self.state),
            "S2: {X2: (G2, G6, G7)}, S4: {X4: (G4)}, S5: {X5: (G5, G8, G9)}"
        )

        self.state.speciation("S2")
        self.assertEqual(
            str(self.state),
            "S4: {X4: (G4)}, S5: {X5: (G5, G8, G9)}, "
            "S6: {X6: (G10, G11, G12)}, S7: {X7: (G13, G14, G15)}"
        )

        self.state.speciation("S4")
        self.assertEqual(
            str(self.state),
            "S5: {X5: (G5, G8, G9)}, S6: {X6: (G10, G11, G12)}, "
            "S7: {X7: (G13, G14, G15)}, S8: {X8: (G16)}, S9: {X9: (G17)}"
        )

        self.state.species = {}

        with self.assertRaisesRegex(
            InvalidEventError,
            "needs at least one species",
        ):
            self.state.speciation()

    def test_extinction(self):
        self.state.speciation()
        self.assertEqual(str(self.state), "S2: {X2: (G2)}, S3: {X3: (G3)}")

        self.state.speciation("S3")
        self.assertEqual(
            str(self.state),
            "S2: {X2: (G2)}, S4: {X4: (G4)}, S5: {X5: (G5)}"
        )

        self.state.extinction("S4")
        self.assertEqual(str(self.state), "S2: {X2: (G2)}, S5: {X5: (G5)}")

        self.state.extinction("S5")
        self.assertEqual(str(self.state), "S2: {X2: (G2)}")

        self.state.extinction()
        self.assertEqual(str(self.state), "")

        with self.assertRaisesRegex(
            InvalidEventError,
            "needs at least one species",
        ):
            self.state.extinction()

    def test_duplication(self):
        self.state.duplication()
        self.assertEqual(str(self.state), "S1: {X2: (G2), X3: (G3)}")

        g4, g5 = Gene(), Gene()
        self.state["S1"]["X2"][2:] = [g4, g5]
        self.assertEqual(str(self.state), "S1: {X2: (G2, G4, G5), X3: (G3)}")

        self.state.duplication("S1", "X3")
        self.assertEqual(
            str(self.state),
            "S1: {X2: (G2, G4, G5), X4: (G6), X5: (G7)}"
        )

        self.state.duplication("S1", "X2", 1, 3)
        self.assertEqual(
            str(self.state),
            "S1: {X4: (G6), X5: (G7), X6: (G2, G8, G9), X7: (G10, G11)}"
        )

        with self.assertRaisesRegex(
            InvalidEventError,
            "cannot copy empty segment",
        ):
            self.state.duplication("S1", "X4", 1, 1)

        with self.assertRaisesRegex(
            InvalidEventError,
            "cannot copy invalid segment",
        ):
            self.state.duplication("S1", "X4", 1, 12)

        self.state.species = {}

        with self.assertRaisesRegex(
            InvalidEventError,
            "needs at least one species",
        ):
            self.state.duplication()

    def test_transfer(self):
        with self.assertRaisesRegex(
            InvalidEventError,
            "needs at least two distinct species",
        ):
            self.state.transfer()

        g2, g3 = Gene(), Gene()
        self.state["S1"]["X1"][2:] = [g2, g3]
        self.assertEqual(str(self.state), "S1: {X1: (G1, G2, G3)}")

        self.state.speciation()
        self.assertEqual(
            str(self.state),
            "S2: {X2: (G4, G5, G6)}, S3: {X3: (G7, G8, G9)}",
        )

        self.state.transfer("S3", "S2", "X3", 0, 2)
        self.assertEqual(
            str(self.state),
            "S2: {X2: (G4, G5, G6), X5: (G12, G13)}, S3: {X4: (G10, G11, G9)}",
        )

        with self.assertRaisesRegex(
            InvalidEventError,
            "cannot transfer to the same species",
        ):
            self.state.transfer("S3", "S3", "X3", 0, 2)

    def test_gain(self):
        self.state.gain(None, None, 1, True)
        self.assertEqual(str(self.state), "S1: {X1: (G1, G2)}")

        self.state.gain(None, None, 1, True)
        self.assertEqual(str(self.state), "S1: {X1: (G1, G3, G2)}")

        self.state.gain(None, None, 0, False)
        self.assertEqual(str(self.state), "S1: {X1: (-G4, G1, G3, G2)}")

        self.state.gain(None, None, 5, False)
        self.assertEqual(str(self.state), "S1: {X1: (-G4, G1, G3, G2, -G5)}")

        self.state.duplication("S1", "X1", 0, 2)
        self.assertEqual(
            str(self.state),
            "S1: {X2: (-G6, G7, G3, G2, -G5), X3: (-G8, G9)}",
        )

        self.state.speciation()
        self.assertEqual(
            str(self.state),
            "S2: {X4: (-G10, G11, G12, G13, -G14), X5: (-G15, G16)}, "
            "S3: {X6: (-G17, G18, G19, G20, -G21), X7: (-G22, G23)}",
        )

        self.state.gain("S3", "X6", 2, True)
        self.assertEqual(
            str(self.state),
            "S2: {X4: (-G10, G11, G12, G13, -G14), X5: (-G15, G16)}, "
            "S3: {X6: (-G17, G18, G24, G19, G20, -G21), X7: (-G22, G23)}",
        )

        self.state.species = {}

        with self.assertRaisesRegex(
            InvalidEventError,
            "needs at least one species",
        ):
            self.state.gain()

    def test_loss(self):
        g2, g3 = Gene(), Gene()
        self.state["S1"]["X1"][2:] = [g2, g3]
        self.assertEqual(str(self.state), "S1: {X1: (G1, G2, G3)}")

        self.state.duplication()
        self.assertEqual(
            str(self.state),
            "S1: {X2: (G4, G5, G6), X3: (G7, G8, G9)}",
        )

        self.state.speciation()
        self.assertEqual(
            str(self.state),
            "S2: {X4: (G10, G11, G12), X5: (G13, G14, G15)}, "
            "S3: {X6: (G16, G17, G18), X7: (G19, G20, G21)}",
        )

        self.state.loss("S3", "X6", 1, 2)
        self.assertEqual(
            str(self.state),
            "S2: {X4: (G10, G11, G12), X5: (G13, G14, G15)}, "
            "S3: {X6: (G16, G18), X7: (G19, G20, G21)}",
        )

        self.state.loss("S2", "X5", 1, 3)
        self.assertEqual(
            str(self.state),
            "S2: {X4: (G10, G11, G12), X5: (G13)}, "
            "S3: {X6: (G16, G18), X7: (G19, G20, G21)}",
        )

        self.state.loss("S3", "X7", 0, 3)
        self.assertEqual(
            str(self.state),
            "S2: {X4: (G10, G11, G12), X5: (G13)}, "
            "S3: {X6: (G16, G18)}",
        )

        self.state.loss("S3", "X6", 0, 2)
        self.assertEqual(str(self.state), "S2: {X4: (G10, G11, G12), X5: (G13)}")

        self.state.species = {}

        with self.assertRaisesRegex(
            InvalidEventError,
            "needs at least one species",
        ):
            self.state.loss()

    def test_cut(self):
        with self.assertRaisesRegex(
            InvalidEventError,
            "needs at least one species with a synteny of length at least 2",
        ):
            self.state.cut()

        g2, g3 = Gene(), Gene()
        self.state["S1"]["X1"][2:] = [g2, g3]
        self.assertEqual(str(self.state), "S1: {X1: (G1, G2, G3)}")

        self.state.duplication("S1", "X1", 2, 3)
        self.assertEqual(str(self.state), "S1: {X2: (G1, G2, G4), X3: (G5)}")

        self.state.speciation()
        self.state.loss("S3", "X6", 0, 3)
        self.assertEqual(
            str(self.state),
            "S2: {X4: (G6, G7, G8), X5: (G9)}, "
            "S3: {X7: (G13)}"
        )

        with self.assertRaisesRegex(
            InvalidEventError,
            "cannot cut at start or end of synteny",
        ):
            self.state.cut("S2", "X4", 0)

        with self.assertRaisesRegex(
            InvalidEventError,
            "cannot cut at start or end of synteny",
        ):
            self.state.cut("S2", "X4", 3)

        with self.assertRaisesRegex(
            InvalidEventError,
            "needs a species with a synteny of length at least 2",
        ):
            self.state.cut("S3")

        with self.assertRaisesRegex(
            InvalidEventError,
            "needs a synteny of length at least 2",
        ):
            self.state.cut("S2", "X5")

        self.state.cut("S2", "X4", 2)
        self.assertEqual(
            str(self.state),
            "S2: {X5: (G9), X8: (G6, G7), X9: (G8)}, "
            "S3: {X7: (G13)}"
        )

    def test_join(self):
        with self.assertRaisesRegex(
            InvalidEventError,
            "needs at least two distinct syntenies",
        ):
            self.state.join()

        g2, g3, g4, g5 = Gene(), Gene(), Gene(), Gene()
        self.state["S1"]["X1"][2:] = [-g2, -g3]
        self.state["S1"].add(Synteny.of(g4, -g5))
        self.assertEqual(
            str(self.state),
            "S1: {X1: (G1, -G2, -G3), X2: (G4, -G5)}",
        )

        self.state.join("S1", "X1", True, "X2", True)
        self.assertEqual(
            str(self.state),
            "S1: {X3: (G1, -G2, -G3, G4, -G5)}",
        )

        self.state.cut("S1", "X3", 3)
        self.state.join("S1", "X4", True, "X5", False)
        self.assertEqual(
            str(self.state),
            "S1: {X6: (G1, -G2, -G3, G5, -G4)}",
        )

        self.state.cut("S1", "X6", 3)
        self.state.join("S1", "X8", False, "X7", False)
        self.assertEqual(
            str(self.state),
            "S1: {X9: (G4, -G5, G3, G2, -G1)}",
        )

        self.state.species = {}

        with self.assertRaisesRegex(
            InvalidEventError,
            "needs at least one species",
        ):
            self.state.join()

    def test_event(self):
        self.state.event(Event.Speciation)
        self.assertEqual(str(self.state), "S2: {X2: (G2)}, S3: {X3: (G3)}")

        self.state.event(Event.Duplication, "S2", "X2")
        self.assertEqual(
            str(self.state),
            "S2: {X4: (G4), X5: (G5)}, S3: {X3: (G3)}"
        )

        self.state.event(Event.Extinction, "S3")
        self.assertEqual(str(self.state), "S2: {X4: (G4), X5: (G5)}")

        self.state.event(Event.Speciation)
        self.assertEqual(
            str(self.state),
            "S4: {X6: (G6), X7: (G7)}, "
            "S5: {X8: (G8), X9: (G9)}"
        )

        self.state.event(Event.Transfer, "S4", "S5", "X6")
        self.assertEqual(
            str(self.state),
            "S4: {X7: (G7), X10: (G10)}, "
            "S5: {X8: (G8), X9: (G9), X11: (G11)}"
        )

        self.state.event(Event.Gain, "S4", "X7", 1, False)
        self.assertEqual(
            str(self.state),
            "S4: {X7: (G7, -G12), X10: (G10)}, "
            "S5: {X8: (G8), X9: (G9), X11: (G11)}"
        )

        self.state.event(Event.Loss, "S5", "X9")
        self.assertEqual(
            str(self.state),
            "S4: {X7: (G7, -G12), X10: (G10)}, "
            "S5: {X8: (G8), X11: (G11)}"
        )

        self.state.event(Event.Cut, "S4", "X7", 1)
        self.assertEqual(
            str(self.state),
            "S4: {X10: (G10), X12: (G7), X13: (-G12)}, "
            "S5: {X8: (G8), X11: (G11)}"
        )

        self.state.event(Event.Join, "S4", "X10", True, "X13", False)
        self.assertEqual(
            str(self.state),
            "S4: {X12: (G7), X14: (G10, G12)}, "
            "S5: {X8: (G8), X11: (G11)}"
        )

        self.state.extinction()
        self.state.extinction()

        with self.assertRaisesRegex(
            InvalidEventError,
            "no event applicable",
        ):
            self.state.event()


    def test_event_prob(self):
        generator = self.generator

        for _ in range(100):
            self.setUp()
            self.state.generator = generator
            self.state.event({
                Event.Gain: 1,
                Event.Loss: 1,
            })

            self.assertIn(
                str(self.state),
                (
                    "S1: {X1: (G1, G2)}",
                    "S1: {X1: (G1, -G2)}",
                    "S1: {X1: (G2, G1)}",
                    "S1: {X1: (-G2, G1)}",
                    "",
                )
            )
