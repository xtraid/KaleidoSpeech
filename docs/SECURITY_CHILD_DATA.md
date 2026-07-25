# Child-data security review

Status: engineering baseline; legal/DPO and clinical approval remain required.

The service pseudonymizes clients, records guardian consent, stores hashes
instead of raw repetition audio, and writes append-only clinical audit events.
Automated output is explicitly non-clinical. Decision training logs contain
signals and model versions, not PCM.

Release controls:

- obtain verifiable guardian consent and support withdrawal;
- define jurisdiction-specific lawful basis, retention and deletion schedules;
- keep identity mapping outside this service with separate access controls;
- encrypt transport and storage, rotate secrets, and restrict reviewer APIs;
- prohibit raw audio and direct identifiers in application logs;
- audit administrative reads, annotations, exports and deletions;
- document processors, data locations, incident response and breach notice;
- perform a DPIA/child-safety review before production;
- test backup deletion and consent-withdrawal propagation;
- require a human professional for clinical interpretation.

`WEBSOCKET_AUTH_TOKEN` is only an integration-level shared-secret mechanism.
Production authorization should use short-lived, audience-bound identities and
per-client/clinician policy at the gateway. The review API is disabled until
`ADMIN_API_TOKEN` is set.
