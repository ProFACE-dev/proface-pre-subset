<!--
SPDX-FileCopyrightText: 2026 Lorenzo Rusnati
SPDX-FileCopyrightText: 2026 Stefano Miccoli

SPDX-License-Identifier: MIT
-->

# ProFACE preprocessor subset

This is a `transforms` plugin for `proface-pre`

## Basic usage

Append this configuration snippet to the ProFACE preprocessor `.toml` job file:

```toml
[[transforms]]
_plugin = "subset"
subdomain = "DOMAIN"
internal_interface = "INTERNAL_INTERFACE"
```

where the key

- `subdomain` must be set to element set to keep, and
- `internal_interface` to the name of a new node set that will contain the interface nodes between the subdomain and the rest of the model.
  This setting is optional: if missing the name `"INTERNAL_INTERFACE"` will be used.
