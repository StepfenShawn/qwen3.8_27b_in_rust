pub struct Sampler {
    state: u64,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    presence_penalty: f32,
    presence: Vec<u8>,
}

impl Sampler {
    pub fn new(
        seed: u64,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        presence_penalty: f32,
        vocab: usize,
    ) -> Self {
        Sampler {
            state: if seed != 0 {
                seed
            } else {
                0x9e37_79b9_7f4a_7c15
            },
            temperature,
            top_k,
            top_p,
            presence_penalty,
            presence: vec![0u8; vocab],
        }
    }

    #[inline]
    fn random(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(2685821657736338717)
    }

    pub fn observe(&mut self, token: u32) {
        if let Some(slot) = self.presence.get_mut(token as usize) {
            *slot = 1;
        }
    }

    #[inline]
    fn adjusted_logit(&self, logits: &[f32], token: u32) -> f32 {
        let value = logits[token as usize];
        if self.presence_penalty == 0.0
            || token as usize >= self.presence.len()
            || self.presence[token as usize] == 0
        {
            value
        } else {
            value - self.presence_penalty
        }
    }

    #[inline]
    fn argmax(&self, logits: &[f32], vocab: usize) -> u32 {
        let mut best = 0u32;
        for id in 1..vocab as u32 {
            if self.adjusted_logit(logits, id) > self.adjusted_logit(logits, best) {
                best = id;
            }
        }
        best
    }

    #[inline]
    fn top_k_candidates(&self, logits: &[f32], vocab: usize, count: usize) -> Vec<(u32, f32)> {
        let mut candidates = vec![(0u32, f32::NEG_INFINITY); count];
        for id in 0..vocab as u32 {
            let logit = self.adjusted_logit(logits, id);
            if logit > candidates[count - 1].1 {
                let mut slot = count - 1;
                while slot > 0 && logit > candidates[slot - 1].1 {
                    candidates[slot] = candidates[slot - 1];
                    slot -= 1;
                }
                candidates[slot] = (id, logit);
            }
        }
        candidates
    }

    #[inline]
    fn softmax(&self, candidates: &[(u32, f32)]) -> (Vec<f32>, f32) {
        let mx_logit = candidates[0].1;
        let mut total = 0.;
        let mut probs = Vec::with_capacity(candidates.len());
        for &(_, logit) in candidates {
            let prob = ((logit - mx_logit) / self.temperature).exp();
            probs.push(prob);
            total += prob;
        }
        (probs, total)
    }

    #[inline]
    fn top_p_cutoff(&self, probs: &[f32], total: f32) -> usize {
        let top_p = if self.top_p > 0.0 && self.top_p < 1.0 {
            self.top_p
        } else {
            1.0
        };
        let mut cumulative = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p / total;
            if cumulative >= top_p {
                return i + 1;
            }
        }
        probs.len()
    }

    #[inline]
    fn sample_from_probs(&mut self, probs: &[f32]) -> usize {
        let total: f32 = probs.iter().sum();
        let unit = (self.random() >> 11) as f64 * 2f64.powi(-53); // random number in [0, 1)
        let target = unit as f32 * total;
        let mut running = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            running += p;
            if running > target {
                return i;
            }
        }
        probs.len() - 1
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let vocab = logits.len();
        if self.temperature <= 0.0 || self.top_k <= 1 {
            return self.argmax(logits, vocab);
        }

        let count = (self.top_k as usize).min(vocab);
        let candidates = self.top_k_candidates(logits, vocab, count);
        let (probs, total) = self.softmax(&candidates);

        let retained = self.top_p_cutoff(&probs, total);
        let picked = self.sample_from_probs(&probs[..retained]);
        candidates[picked].0
    }
}
