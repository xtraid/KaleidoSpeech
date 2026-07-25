# MVP pediatrico e streaming — report tecnico

## Risultato

Il branch contiene ora un percorso sperimentale per esercizi di pronuncia
inglese supervisionati da un logopedista. Il sistema propone una frase,
acquisisce audio PCM mono 16 kHz in frame da 40 ms, restituisce osservazioni
progressive e conserva ripetizioni/revisioni in SQLite.

È un ausilio sperimentale alla raccolta e revisione, non un dispositivo
diagnostico: non formula diagnosi né prescrive terapia.

## Confine decisionale

Il dataset attuale è adulto e sbilanciato. I test precedenti hanno mostrato
falsi rifiuti su una voce femminile e ambiguità con parole foneticamente
vicine. La pipeline separa quindi:

1. feedback sicuro in tempo reale: presenza di voce, RMS, picco e clipping;
2. distanza acustica provvisoria dai benchmark DTW;
3. verdetto clinico, inserito esclusivamente dal logopedista.

Il software salva solamente `REVIEW_REQUIRED` o `RETRY`. I valori `accepted`,
`speech_error` e `retry` appartengono alla revisione umana.

## Streaming

- 16.000 Hz, mono, PCM signed 16-bit little-endian;
- 640 campioni / 1.280 byte per frame;
- un frame ogni 40 ms;
- osservazione VAD/qualità su ciascun frame;
- aggiornamento DTW ogni 200 ms;
- finestra mobile DTW di 1.600 ms;
- limite predefinito: 15 secondi.

Il DTW non gira ogni 40 ms perché potrebbe bloccare l'acquisizione. È eseguito
da un worker a capacità uno: la cattura non attende il risultato e non accumula
job obsoleti. Ogni frame produce comunque un'inferenza leggera e riporta
l'evidenza acustica completa più recente.

Sul WAV locale di controllo (51 frame), dopo lo spostamento asincrono, la
chiamata di acquisizione ha impiegato in media 0,20 ms e al massimo 0,75 ms:
ampiamente sotto il budget di 40 ms. Il DTW resta sperimentale e va profilato
di nuovo sull'hardware di destinazione e con più parole/prototipi.

Una frase continua non ha confini di parola affidabili ricavabili dai soli
benchmark di parole isolate. `alignment_status` resta quindi `not_available`:
prima del feedback per fonema serve un forced aligner inglese validato su
parlato pediatrico.

### Uso

```bash
SQLITE_PATH=/tmp/advx-demo.sqlite3 \
  uv run python -m scripts.streaming_demo \
  bird follow forward happy learn --duration 8
```

Il programma mostra la frase, attende Invio, registra e stampa eventi JSON.
L'API offre anche `/streaming/sessions/{session_id}`: comando `session.start`,
frame binari da 1.280 byte e comando `session.finish`.

## Persistenza clinica minima

Le nuove tabelle gestiscono minore pseudonimizzato e fascia d'età, consenso
del tutore, esercizio suono/sillaba/parola/frase, sessioni, ripetizioni,
revisioni e audit append-only. SQLite non conserva il PCM: salva SHA-256,
durata e osservazioni. L'eventuale archivio audio dovrà essere separato,
cifrato e soggetto a retention e cancellazione dopo revoca.

## Redis 8 e vettori

La v3 usa un solo runtime `redis:8.8.0-alpine`. Redis 8 integra Redis Search,
supporta indici HNSW su hash/JSON e include i Vector Set. Il progetto seleziona
l'opzione di licenza open-source AGPLv3; gli obblighi copyleft devono essere
valutati prima della distribuzione.

`app/vector_index.py` indicizza gli embedding riassuntivi a 26 dimensioni.
SQLite resta autorevole per metadati, soglie, audit e revisioni; Redis serve
per cache, stream e retrieval dei vicini. `scripts.check_redis` verifica
raggiungibilità, `FT.SEARCH` e `VSIM` senza modificare lo stato.

Un database vettoriale accelera il retrieval, ma non corregge bias o errori
fonetici.

- https://redis.io/docs/latest/develop/whats-new/8-0/
- https://redis.io/docs/latest/develop/interact/search-and-query/query/vector-search/
- https://redis.io/legal/licenses/

## Dataset da valutare

Priorità ai corpora pediatrici e annotati:

- PERCEPT-R: errori rotici pediatrici;
- PERCEPT-US: bambini con speech sound disorder;
- UltraSuite: articolazione e ultrasuoni pediatrici;
- SEED/SPROUT: sviluppo fonologico infantile;
- CSLU Kids: parlato infantile;
- SpeechOcean762: pronuncia inglese non madrelingua.

Fonti:

- https://www.isca-archive.org/interspeech_2022/benway22_interspeech.html
- https://www.isca-archive.org/interspeech_2025/eads25_interspeech.html
- https://ultrasuite.github.io/
- https://pedzstarlab.soc.northwestern.edu/datasets/
- https://catalog.ldc.upenn.edu/LDC2007S18
- https://www.openslr.org/101/

“Gratis da scaricare” non implica uso commerciale o redistribuzione. Prima
dell'import serve una scheda per licenza, consenso, età, finalità consentite e
attribuzione.

## Rischi e gate clinici

- Il benchmark non rappresenta ancora età, genere, accento o disturbo.
- Una distanza acustica non equivale a correttezza articolatoria.
- La voce di un minore richiede DPIA, controllo accessi, cifratura e revoca.
- Se il prodotto formula valutazioni cliniche va verificata la disciplina del
  software medicale.
- Latenza DTW e perdita frame vanno misurate sul dispositivo target.

Prima di provare il sistema con bambini servono: protocollo approvato da
logopedisti, consenso e DPIA, corpus pediatrico legalmente utilizzabile, split
per bambino, metriche per sottogruppo ed errore, calibrazione dei falsi esiti,
forced alignment validato o confini revisionati manualmente, test di rumore e
latenza, e UI che distingua osservazione automatica e giudizio clinico.

Riferimenti:

- ASHA: https://www.asha.org/practice-portal/clinical-topics/articulation-and-phonology/
- UE medical devices: https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en
- GDPR: https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng
