//! MT19937 (Mersenne Twister) with Python `random.Random(seed)` compatibility,
//! used so season simulations produce identical draws to the Flask app for the
//! same seed (seed contract in PORT_PLAN §6.5 / M4 acceptance criteria).
//!
//! CPython seeds integer args via `init_by_array([key], 1)` (the
//! int→key-array path), and `random()` returns `genrand_res53()` (53-bit
//! mantissa from two tempered draws); `randrange(n)` is `floor(random() * n)`.

pub struct Mt19937 {
    mt: [u32; 624],
    mti: usize,
}

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;

impl Mt19937 {
    /// `init_by_array([seed], 1)` — matches CPython `random.Random(seed)`.
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; N];
        // init_genrand(19650218)
        mt[0] = 19650218;
        for i in 1..N {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        // init_by_array with a single 32-bit key.
        let key = [seed];
        let mut i = 1usize;
        let mut j = 0usize;
        let k = N.max(key.len());
        for _ in 0..k {
            mt[i] = (mt[i]
                ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525)))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            mt[i] = (mt[i]
                ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }
        mt[0] = UPPER_MASK;
        Self { mt, mti: N }
    }

    fn twist(&mut self) {
        for i in 0..N {
            let y = (self.mt[i] & UPPER_MASK) | (self.mt[(i + 1) % N] & LOWER_MASK);
            let mag = if y & 1 == 1 { MATRIX_A } else { 0 };
            self.mt[i] = self.mt[(i + M) % N] ^ (y >> 1) ^ mag;
        }
        self.mti = 0;
    }

    /// Python `random.Random.random()` — [0, 1) with 53-bit resolution.
    /// Matches CPython's `genrand_res53`: two tempered draws combined.
    pub fn next_f64(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// Python `random.Random.randrange(n)` — via `_randbelow` (getrandbits +
    /// rejection), matching CPython exactly.
    pub fn randrange(&mut self, n: usize) -> usize {
        self.randbelow(n)
    }

    /// `random.Random.randint(a, b)` (inclusive).
    pub fn randint(&mut self, a: usize, b: usize) -> usize {
        if b < a {
            return a;
        }
        a + self.randbelow(b - a + 1)
    }

    /// `random._randbelow(n)` with getrandbits (rejection sampling).
    pub fn randbelow(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let k = (usize::BITS - n.leading_zeros()) as usize; // bit_length(n)
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `random.getrandbits(k)`: high-order bits of 32-bit tempered words.
    pub fn getrandbits(&mut self, k: usize) -> usize {
        if k == 0 {
            return 0;
        }
        if k <= 32 {
            return (self.genrand_u32() >> (32 - k)) as usize;
        }
        let mut r: usize = 0;
        let mut bits = 0usize;
        while bits < k {
            let take = 32usize.min(k - bits);
            r = (r << take) | (self.genrand_u32() >> (32 - take)) as usize;
            bits += take;
        }
        r
    }

    fn genrand_u32(&mut self) -> u32 {
        if self.mti >= N {
            self.twist();
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_random_42() {
        // Python: random.Random(42).random() first 5 draws.
        let mut rng = Mt19937::new(42);
        let expected = [
            0.6394267984578837,
            0.025010755222666936,
            0.27502931836911926,
            0.22321073814882275,
            0.7364712141640124,
        ];
        for e in expected {
            let got = rng.next_f64();
            assert!((got - e).abs() < 1e-15, "got {got} expected {e}");
        }
    }

    #[test]
    fn matches_python_randrange() {
        // Python: [random.Random(7).randrange(10) for _ in range(8)]
        let mut rng = Mt19937::new(7);
        let expected = [5, 2, 6, 0, 1, 8, 1, 5];
        for e in expected {
            assert_eq!(rng.randrange(10), e);
        }
    }
}
