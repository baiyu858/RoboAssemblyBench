# Legacy constraint integration path

The implementation has moved to
[`toolkits/constraint_checking`](../toolkits/constraint_checking/README.md).

Imports under `constraint_integration` remain available as thin compatibility
forwarders. New code should import from `toolkits.constraint_checking.integration`.
