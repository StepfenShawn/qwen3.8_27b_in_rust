pub struct Sampler {
    state: u64,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    presence_penalty: f32,
    presence: Vec<u8>,
}

impl Sampler {
    pub fn new(seed: u64, temperature: f32, top_k: u32, top_p: f32, presence_penalty: f32, vocab: usize) -> Self {
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

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let vocab = logits.len();
        if self.temperature <= 0.0 || self.top_k <= 1 {
            let mut best = 0u32;
            for id in 1..vocab as u32 {
                if self.adjusted_logit(logits, id) > self.adjusted_logit(logits, best) {
                    best = id;
                }
            }
            return best;
        }

        let count = (self.top_k as usize).min(vocab);
        let mut candidates: Vec<(u32, f32)> = vec![(0, f32::NEG_INFINITY); count];
        for id in 0..vocab as u32 {
            let logit = self.adjusted_logit(logits, id);
            if logit <= candidates[count - 1].1 {
                continue;
            }
            let mut slot = count - 1;
            while slot > 0 && logit > candidates[slot - 1].1 {
                candidates[slot] = candidates[slot - 1];
                slot -= 1;
            }
            candidates[slot] = (id, logit);
        }

        let mut probs = vec![0.0f32; count];
        let mut total = 0.0f32;
        for i in 0..count {
            probs[i] = ((candidates[i].1 - candidates[0].1) / self.temperature).exp();
            total += probs[i];
        }

        let top_p = if self.top_p > 0.0 && self.top_p < 1.0 {
            self.top_p
        } else {
            1.0
        };
        let mut cumulative = 0.0f32;
        let mut retained = count;
        for i in 0..count {
            cumulative += probs[i] / total;
            if cumulative >= top_p {
                retained = i + 1;
                break;
            }
        }

        let mut retained_total = 0.0f32;
        for i in 0..retained {
            retained_total += probs[i];
        }
        let unit = (self.random() >> 11) as f64 * 2f64.powi(-53);
        let target = unit as f32 * retained_total;
        let mut running = 0.0f32;
        let mut selected = candidates[retained - 1].0;
        for i in 0..retained {
            running += probs[i];
            if target < running {
                selected = candidates[i].0;
                break;
            }
        }
        selected
    }
}