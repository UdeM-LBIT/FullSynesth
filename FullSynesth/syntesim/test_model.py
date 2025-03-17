from .model import Gene, Synteny, Species
from unittest import TestCase


class TestGene(TestCase):
    def test_ids(self):
        a, b, c = Gene(), Gene(), Gene()
        self.assertNotEqual(a.id, b.id)
        self.assertNotEqual(a.id, c.id)
        self.assertNotEqual(b.id, c.id)

    def test_neg(self):
        a = Gene()
        b = -a
        c = -b
        self.assertEqual(a.id, b.id)
        self.assertEqual(b.id, c.id)
        self.assertTrue(a.orientation)
        self.assertFalse(b.orientation)
        self.assertTrue(c.orientation)

    def test_clone(self):
        a = Gene()
        b = a.clone()
        c = (-a).clone()
        self.assertNotEqual(a.id, b.id)
        self.assertNotEqual(a.id, c.id)
        self.assertNotEqual(b.id, c.id)
        self.assertTrue(b.orientation)
        self.assertFalse(c.orientation)

    def test_serialize(self):
        a = Gene()
        b = -a
        self.assertEqual("-" + str(a), str(b))
        self.assertEqual(a.serialize(), (a.id, True))
        self.assertEqual(b.serialize(), (a.id, False))
        self.assertEqual(Gene.unserialize(b.serialize()), b)


class TestSynteny(TestCase):
    def test_ids(self):
        a = Gene()
        x, y, z = Synteny.of(a), Synteny.of(a), Synteny.of(a)
        self.assertNotEqual(x.id, y.id)
        self.assertNotEqual(x.id, z.id)
        self.assertNotEqual(y.id, z.id)

    def test_nonempty(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "synteny must have at least one gene",
        ):
            x = Synteny()

    def test_add(self):
        a, b, c = Gene(), Gene(), Gene()
        s1 = Synteny.of(a, b, c)

        d, e, f = Gene(), Gene(), Gene()
        s2 = Synteny.of(d, e, f)

        s3 = s1 + s2
        s4 = s1 + -s2
        s5 = -s1 + s2
        s6 = -s1 + -s2

        self.assertEqual(s3.genes, [a, b, c, d, e, f])
        self.assertEqual(s4.genes, [a, b, c, -f, -e, -d])
        self.assertEqual(s5.genes, [-c, -b, -a, d, e, f])
        self.assertEqual(s6.genes, [-c, -b, -a, -f, -e, -d])

    def test_get(self):
        a, b, c, d, e = Gene(), Gene(), Gene(), Gene(), Gene()
        s = Synteny.of(a, b, -c, d, e)

        self.assertEqual(s[0], a)
        self.assertEqual(s[1], b)
        self.assertEqual(s[2], -c)
        self.assertEqual(s[3], d)
        self.assertEqual(s[4], e)

    def test_slice(self):
        a, b, c, d, e = Gene(), Gene(), Gene(), Gene(), Gene()
        s = Synteny.of(a, b, -c, d, e)

        self.assertEqual(s[1:3].id, s.id)
        self.assertEqual(s[1:3].genes, [b, -c])
        self.assertEqual(s[:3].genes, [a, b, -c])
        self.assertEqual(s[3:].genes, [d, e])
        self.assertEqual(s[:].genes, [a, b, -c, d, e])
        self.assertEqual(s[2:0:-1].genes, [c, -b])
        self.assertEqual(s[::-1].genes, [-e, -d, c, -b, -a])
        self.assertEqual(-s, s[::-1])

    def test_del(self):
        a, b, c, d, e = Gene(), Gene(), Gene(), Gene(), Gene()
        s = Synteny.of(a, b, -c, d, e)
        orig_id = s.id

        del s[2:4]
        self.assertEqual(s.id, orig_id)
        self.assertEqual(s.genes, [a, b, e])

        del s[:]
        self.assertEqual(s.id, orig_id)
        self.assertEqual(s.genes, [])

    def test_set(self):
        a, b, c, d, e = Gene(), Gene(), Gene(), Gene(), Gene()
        s = Synteny.of(a, b, -c, d, e)
        orig_id = s.id

        f, g = Gene(), Gene()
        s[2:3] = [f, -g]

        self.assertEqual(s.id, orig_id)
        self.assertEqual(s.genes, [a, b, f, -g, d, e])

        s[5:5] = [-c]
        self.assertEqual(s.id, orig_id)
        self.assertEqual(s.genes, [a, b, f, -g, d, -c, e])

        s[3] = g
        self.assertEqual(s.id, orig_id)
        self.assertEqual(s.genes, [a, b, f, g, d, -c, e])

        with self.assertRaisesRegex(TypeError, "can only assign a single gene"):
            s[3] = [g]

        with self.assertRaisesRegex(TypeError, "can only assign an iterable"):
            s[3:5] = g

        s[2:5] = s[4:1:-1]
        self.assertEqual(s.genes, [a, b, -d, -g, -f, -c, e])

    def test_clone(self):
        a, b, c, d, e = Gene(), Gene(), Gene(), Gene(), Gene()
        s = Synteny.of(a, b, -c, d, e)
        s1 = s[:3].clone()
        s2 = s[3:].clone()

        self.assertNotEqual(s.id, s1.id)
        self.assertNotEqual(s.id, s2.id)
        self.assertNotEqual(s1.id, s2.id)

        for i in range(len(s1)):
            self.assertEqual(s1[i], s[i])

        for i in range(len(s2)):
            self.assertEqual(s2[i], s[i + len(s1)])

    def test_deep_clone(self):
        a, b, c, d, e = Gene(), Gene(), Gene(), Gene(), Gene()
        s = Synteny.of(a, -b, c, d, -e)
        sc = s.deep_clone()

        self.assertNotEqual(sc.id, s.id)
        self.assertEqual(len(s), len(sc))

        for i in range(len(s)):
            self.assertNotEqual(s[i], sc[i])
            self.assertEqual(s[i].orientation, sc[i].orientation)

    def test_serialize(self):
        a, b, c, d, e = Gene(), Gene(), Gene(), Gene(), Gene()
        s = Synteny.of(a, b, -c, d, e)
        self.assertEqual(
            str(s),
            f"{s.id}: ({str(a)}, {str(b)}, {str(-c)}, {str(d)}, {str(e)})"
        )
        self.assertEqual(
            s.serialize(),
            [
                a.serialize(), b.serialize(), (-c).serialize(),
                d.serialize(), e.serialize(),
            ]
        )
        self.assertEqual(
            Synteny.unserialize(s.serialize()).genes,
            s.genes,
        )


class TestSpecies(TestCase):
    def test_ids(self):
        a = Gene()
        x = Synteny.of(a)
        s = Species.of(x)
        t = Species.of(x)
        u = Species.of(x)
        self.assertNotEqual(s.id, t.id)
        self.assertNotEqual(s.id, u.id)
        self.assertNotEqual(t.id, u.id)

    def test_nonempty(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "species must have at least one synteny",
        ):
            x = Species()

    def test_add_remove(self):
        a, b, c = Gene(), Gene(), Gene()
        s1 = Synteny.of(a, b, c)

        d, e, f = Gene(), Gene(), Gene()
        s2 = Synteny.of(d, e, f)

        sp = Species.of(s1)
        self.assertEqual(sp.syntenies, {s1.id: s1})

        sp.add(s2)
        self.assertEqual(sp.syntenies, {s1.id: s1, s2.id: s2})

        sp.remove(s1)
        self.assertEqual(sp.syntenies, {s2.id: s2})

        sp.remove(s2)
        self.assertEqual(sp.syntenies, {})

    def test_clone(self):
        a, b, c = Gene(), Gene(), Gene()
        s1 = Synteny.of(a, b, c)

        d, e, f = Gene(), Gene(), Gene()
        s2 = Synteny.of(d, e, f)

        sp = Species.of(s1, s2)
        spc = sp.clone()

        self.assertNotEqual(sp.id, spc.id)
        self.assertEqual(sp.syntenies, spc.syntenies)

    def test_deep_clone(self):
        a, b, c, d = Gene(), Gene(), Gene(), Gene()
        s1 = Synteny.of(a, -b, c, d)

        e, f = Gene(), Gene()
        s2 = Synteny.of(e, -f)

        sp = Species.of(s1, s2)
        spc = sp.deep_clone()

        self.assertNotEqual(sp.id, spc.id)
        self.assertEqual(len(sp), len(spc))

        for i in sp.syntenies.keys():
            for j in spc.syntenies.keys():
                self.assertNotEqual(i, j)

        for i in sp.syntenies.keys():
            self.assertTrue(any(
                len(sp[i]) == len(spc[j])
                and all(
                    sp[i][k] != spc[j][k]
                    and sp[i][k].orientation == spc[j][k].orientation
                    for k in range(len(sp[i]))
                )
                for j in spc.syntenies.keys()
            ))

    def test_serialize(self):
        a, b, c, d, e, f, g, h = Gene(), Gene(), Gene(), Gene(), Gene(), \
            Gene(), Gene(), Gene()
        s1 = Synteny.of(a, -b, c, d)
        s2 = Synteny.of(e, f, g)
        s3 = Synteny.of(-h)

        sp = Species.of(s1, s2, s3)
        self.assertEqual(
            str(sp),
            f"{sp.id}: {{{str(s1)}, {str(s2)}, {str(s3)}}}"
        )
        self.assertEqual(
            sp.serialize(),
            {
                s1.id: s1.serialize(),
                s2.id: s2.serialize(),
                s3.id: s3.serialize(),
            }
        )
        self.assertEqual(
            Species.unserialize(sp.serialize()).syntenies,
            sp.syntenies,
        )
