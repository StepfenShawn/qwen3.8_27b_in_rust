use crate::gguf::Gguf;
use crate::kernel::Q38Iq1sRepack ;

impl Q38Iq1sRepack for Gguf {
    fn prepare_iq1_s_repacks(&mut self) -> super::Q38CoreResult<()> {
        todo!()
    }

    fn release_iq1_s_repacks(&mut self) {
        todo!()
    }
}
