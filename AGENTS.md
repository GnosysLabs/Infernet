User has ADHD. Please make small incremental changes that show immediate feedback and increment in this way. It will help with his attention.

Inference placement is a product invariant:

- Treat physical-machine identity, not peer or process identity, as the unit of
  placement. Multiple peers on one computer do not satisfy a multi-machine
  plan.
- Whenever two or more eligible distinct physical machines exist, always split
  inference across them. Never select a sole-machine route while another
  eligible physical machine is available.
- Allow sole-machine inference only on the requester's own physical machine and
  only when it is the sole eligible option.
- If the sole eligible physical machine is remote, wait for another eligible
  machine or reject the request; never execute entirely on that remote machine.
- When the requester and a remote physical machine are both eligible, both must
  participate in the split plan.
- Do not add a fallback that silently collapses a failed distributed plan onto
  one remote machine.
