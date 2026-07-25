# Progettazione del correttore di pronuncia

## 1. Obiettivo

advX deve valutare se l'utente:

1. ha pronunciato la parola richiesta;
2. ha realizzato correttamente i fonemi della parola;
3. ha prodotto audio sufficiente per una valutazione affidabile.

Il sistema non deve confondere il riconoscimento della parola con la correzione
della pronuncia. Una parola diversa e una parola corretta pronunciata male sono
due errori distinti e devono produrre motivazioni differenti.

Il risultato non è una diagnosi clinica. In assenza di evidenza sufficiente il
sistema deve restituire `UNDECIDABLE` o `RETRY`, senza trasformare
l'incertezza in un giudizio positivo o negativo.

## 2. Requisiti funzionali

Per una parola obiettivo, per esempio `bird`, il sistema deve distinguere:

| Audio osservato | Risultato | Motivo |
| --- | --- | --- |
| `bird` pronunciata correttamente | `CORRECT` | parola e fonemi compatibili |
| una parola diversa, per esempio `happy` | `INCORRECT` | parola non corrispondente |
| `bird` con uno o più fonemi errati | `INCORRECT` | errore fonetico localizzato |
| audio troppo breve, silenzioso o degradato | `RETRY` | qualità insufficiente |
| evidenza fonetica ambigua | `UNDECIDABLE` | confidenza insufficiente |

Il feedback deve indicare:

- parola richiesta;
- parola riconosciuta, quando affidabile;
- fonemi attesi e fonemi osservati;
- intervallo temporale del possibile errore;
- punteggio e confidenza;
- suggerimento comprensibile per l'utente;
- versione dei modelli e delle soglie utilizzate.

## 3. Principio architetturale

La decisione finale usa più segnali indipendenti. Nessun singolo confronto DTW
può dichiarare da solo che una pronuncia è corretta.

```text
Microfono
    |
    v
Acquisizione PCM 16 kHz
    |
    v
Controllo qualità e segmentazione
    |
    +---- audio insufficiente ----------------------> RETRY
    |
    v
Riconoscimento della parola
    |
    +---- parola diversa con alta confidenza ------> INCORRECT / WRONG_WORD
    |
    v
Allineamento forzato parola-fonemi
    |
    +---- allineamento non affidabile -------------> UNDECIDABLE
    |
    v
Valutazione di ogni fonema
    |
    +---- fonema incompatibile ---------------------> INCORRECT / PHONE_ERROR
    |
    v
Fusione con segnali acustici secondari
    |
    v
CORRECT oppure UNDECIDABLE
```

## 4. Componenti

### 4.1 Audio capture

Responsabilità:

- acquisire PCM mono a 16 kHz, signed int16 little-endian;
- produrre frame da 40 ms;
- conservare numeri di sequenza e timestamp monotoni;
- segnalare overflow e perdita di frame;
- permettere la selezione esplicita del dispositivo di input.

L'applicazione non deve dipendere implicitamente dal microfono predefinito del
sistema. CLI e API devono accettare un identificatore di dispositivo.

### 4.2 Quality gate e voice activity detection

Responsabilità:

- stimare rumore di fondo prima dell'enunciato;
- calcolare una soglia VAD adattiva;
- individuare inizio e fine della parola;
- rimuovere silenzio eccessivo senza tagliare consonanti iniziali o finali;
- misurare clipping, durata, rapporto voce/silenzio e signal-to-noise ratio.

La soglia non deve essere un valore universale. Una possibile inizializzazione è:

```text
voice_threshold = noise_rms_median + K * noise_rms_deviation
```

con limiti minimi e massimi configurabili. La calibrazione deve essere
registrata insieme al tentativo.

### 4.3 Lexicon service

Converte la parola richiesta in una o più pronunce accettabili.

Esempio concettuale:

```json
{
  "word": "bird",
  "locale": "en-US",
  "accepted_pronunciations": [
    ["B", "ER", "D"]
  ],
  "lexicon_version": "en-us-v1"
}
```

Il lessico deve:

- distinguere locale e variante linguistica;
- supportare più pronunce valide;
- conservare stress e durata quando rilevanti;
- non penalizzare automaticamente una variante regionale ammessa.

### 4.4 Word identity gate

Verifica che l'audio contenga la parola richiesta.

Input:

- segmento audio pulito;
- parola obiettivo;
- insieme chiuso delle parole previste dall'esercizio.

Output:

```json
{
  "target_word": "bird",
  "recognized_word": "happy",
  "target_probability": 0.03,
  "recognized_probability": 0.91,
  "status": "MISMATCH",
  "model_version": "word-gate-v1"
}
```

Per esercizi a vocabolario chiuso è preferibile confrontare esplicitamente tutte
le parole candidate. Se una parola diversa vince con margine sufficiente, il
risultato è `WRONG_WORD`. Una bassa confidenza generale non deve diventare un
errore certo: deve produrre `UNDECIDABLE`.

### 4.5 Phoneme acoustic model

Il modello converte l'audio in probabilità temporali sui fonemi. Non restituisce
soltanto una trascrizione finale, ma evidenza per ogni intervallo audio.

Requisiti:

- output frame-level o segment-level;
- fonemi coerenti con il lessico;
- probabilità calibrabili;
- supporto della lingua e del locale dell'esercizio;
- esecuzione separabile dalla logica di business;
- versionamento completo del modello.

Il modello deve essere incapsulato dietro un protocollo, così da poter cambiare
implementazione senza modificare il motore decisionale.

Per l'MVP la scelta concreta è
`facebook/wav2vec2-xlsr-53-espeak-cv-ft`, un wav2vec2/XLS-R multilingue
fine-tuned con obiettivo CTC su etichette fonetiche e distribuito con licenza
Apache-2.0. La revisione è fissata a
`2c733782da5604684829819a5eb744c193fe9398`. L'adapter applicativo è
`TransformersWav2Vec2PhonemeModel`; il modello gira sul server e riceve PCM
16 kHz dell'intera utterance. La revisione dell'artefatto deve essere fissata
prima della calibrazione; un branch mobile come `@main` non è ammesso.

Questa scelta è una baseline tecnica, non una certificazione della sua
accuratezza su bambini o learner speech. Se non supera i gate della sezione 9,
va fine-tunato su dati adeguati o sostituito mantenendo lo stesso contratto.

```python
class PhonemeModel(Protocol):
    def infer(self, pcm_s16le: bytes, sample_rate: int) -> PhonemeEvidence:
        ...
```

### 4.6 Forced aligner

Allinea le probabilità acustiche con la sequenza fonetica attesa.

Per ogni fonema produce:

```json
{
  "expected_phone": "ER",
  "observed_phone": "IH",
  "start_ms": 120,
  "end_ms": 310,
  "expected_probability": 0.22,
  "best_alternative_probability": 0.68,
  "alignment_confidence": 0.84
}
```

L'allineatore deve supportare:

- inserzioni: suoni aggiunti;
- cancellazioni: fonemi mancanti;
- sostituzioni: fonema diverso;
- durate anomale;
- più pronunce valide della stessa parola.

L'MVP usa un allineamento Viterbi sul grafo CTC blank-interleaved, implementato
internamente in NumPy (`numpy-ctc-viterbi-v1`). Non dipende dalle API forced
alignment di torchaudio, ormai deprecate. La confidenza dell'allineamento indica
se lo span contiene evidenza fonetica stabile; è separata dalla probabilità del
fonema atteso, altrimenti una sostituzione reale verrebbe scambiata per un
allineamento non valutabile.

Per parole isolate sotto 400 ms l'allineamento è considerato ad alto rischio:
se ci sono meno frame CTC dei fonemi attesi, uno span è vuoto o la confidenza
scende sotto soglia, il risultato è `UNDECIDABLE`. Il sistema non allunga
artificialmente l'audio per ottenere un verdetto.

### 4.7 Pronunciation scorer

Calcola un punteggio per fonema usando almeno:

- probabilità del fonema atteso;
- probabilità del migliore fonema alternativo;
- qualità dell'allineamento;
- durata rispetto alla distribuzione di riferimento;
- qualità globale dell'audio.

Una forma iniziale del punteggio può essere:

```text
phone_score =
    log P(fonema_atteso | audio)
    - log P(migliore_alternativa | audio)
```

Le soglie devono essere calibrate per fonema e contesto. Una sola soglia globale
penalizzerebbe fonemi naturalmente più difficili da distinguere.

Il risultato contiene sia il punteggio numerico sia una categoria:

```text
PASS | BORDERLINE | FAIL | NOT_SCORABLE
```

### 4.8 Decision engine

Ordine delle regole:

1. qualità insufficiente → `RETRY`;
2. parola diversa con alta confidenza → `INCORRECT`;
3. identità della parola incerta → `UNDECIDABLE`;
4. allineamento non affidabile → `UNDECIDABLE`;
5. almeno un errore fonetico affidabile → `INCORRECT`;
6. tutti i fonemi sufficientemente affidabili → `CORRECT`;
7. ogni altro caso → `UNDECIDABLE`.

Il sistema deve richiedere evidenza positiva per restituire `CORRECT`. La sola
assenza di un errore rilevato non è evidenza di correttezza.

### 4.9 Feedback generator

Traduce il risultato tecnico in un messaggio utile:

```json
{
  "status": "INCORRECT",
  "reason": "PHONE_SUBSTITUTION",
  "target_word": "bird",
  "recognized_word": "bird",
  "phone_errors": [
    {
      "expected": "ER",
      "observed": "IH",
      "start_ms": 120,
      "end_ms": 310
    }
  ],
  "message": "La vocale centrale di “bird” è risultata troppo vicina a /ɪ/."
}
```

I messaggi pedagogici devono essere separati dal modello acustico e revisionati
da una persona competente in fonetica o logopedia.

## 5. Ruolo del DTW esistente

Il DTW attuale non deve essere eliminato, ma declassato a segnale secondario.
Può contribuire a:

- rilevare audio molto lontano dagli esempi della parola;
- confrontare ritmo e durata complessiva;
- scegliere esempi simili per il feedback;
- creare una baseline durante la migrazione.

Non può:

- identificare in modo affidabile il fonema errato;
- distinguere tutte le pronunce scorrette da quelle corrette;
- sostituire un modello fonetico;
- calibrare soglie cliniche usando soltanto parole corrette.

## 6. Contratto del servizio

### Richiesta

```json
{
  "session_id": "session-123",
  "target_word": "bird",
  "locale": "en-US",
  "pcm_encoding": "s16le",
  "sample_rate": 16000
}
```

I frame audio continuano a essere inviati separatamente in blocchi da 40 ms.

### Risposta finale

```json
{
  "status": "INCORRECT",
  "reason_codes": ["PHONE_SUBSTITUTION"],
  "target_word": "bird",
  "recognized_word": "bird",
  "word_identity_confidence": 0.89,
  "pronunciation_score": 0.54,
  "phones": [
    {
      "expected": "B",
      "observed": "B",
      "start_ms": 30,
      "end_ms": 95,
      "score": 0.91,
      "status": "PASS"
    },
    {
      "expected": "ER",
      "observed": "IH",
      "start_ms": 95,
      "end_ms": 310,
      "score": 0.21,
      "status": "FAIL"
    },
    {
      "expected": "D",
      "observed": "D",
      "start_ms": 310,
      "end_ms": 390,
      "score": 0.86,
      "status": "PASS"
    }
  ],
  "quality": {
    "status": "ACCEPT",
    "snr_db": 21.4,
    "clipping": false
  },
  "versions": {
    "phoneme_model": "phoneme-model-v1",
    "lexicon": "en-us-v1",
    "alignment": "ctc-align-v1",
    "calibration": "children-en-us-v1",
    "decision": "pronunciation-decision-v2"
  }
}
```

## 7. Persistenza

Nuove entità consigliate:

### `pronunciation_model_versions`

- nome e versione del modello;
- hash dell'artefatto;
- lingua e locale;
- data di attivazione;
- configurazione di inferenza.

### `phoneme_calibrations`

- fonema;
- contesto sinistro e destro opzionali;
- fascia di parlanti o popolazione;
- soglie `PASS`, `BORDERLINE` e `FAIL`;
- numerosità e metriche del dataset;
- versione.

### `pronunciation_evaluations`

- tentativo e parola obiettivo;
- parola riconosciuta e confidenza;
- risultato finale e motivazioni;
- qualità audio;
- versioni di tutti i componenti;
- timestamp.

### `phoneme_evaluations`

- valutazione di appartenenza;
- fonema atteso e osservato;
- intervallo temporale;
- probabilità e margine;
- punteggio calibrato;
- risultato.

L'audio grezzo non deve essere conservato per impostazione predefinita. Quando
la conservazione è necessaria per ricerca o revisione, deve avere consenso,
retention esplicita, cifratura e controllo degli accessi.

## 8. Dati necessari

Per calibrare realmente il sistema servono:

- pronunce corrette della stessa parola;
- pronunce scorrette etichettate per fonema;
- parole diverse usate come negativi;
- varietà di voce, accento, microfono e rumore;
- separazione dei parlanti tra training, validation e test;
- annotazioni revisionate da esperti;
- un test set congelato che non influenzi le soglie.

Usare altre parole come unici esempi negativi non è sufficiente: insegna al
sistema a distinguere `bird` da `happy`, ma non una buona `/ɜː/` da una vocale
errata dentro `bird`.

## 9. Calibrazione

Le soglie devono essere scelte sul validation set e misurate sul test set.
Metriche minime:

- false acceptance rate di parole diverse;
- false acceptance rate di errori fonetici;
- false rejection rate di pronunce corrette;
- accuratezza per fonema;
- equal error rate;
- percentuale di `UNDECIDABLE`;
- risultati separati per gruppi rilevanti del dataset;
- latenza media e percentile 95.

L'obiettivo non deve essere ridurre artificialmente gli `UNDECIDABLE`. Nei casi
ambigui, l'astensione è un comportamento corretto.

Gate iniziali dell'MVP, misurati sul test set congelato speaker-disjoint:

| Metrica | Limite |
| --- | ---: |
| falsa accettazione di una parola diversa | ≤ 1% |
| falsa accettazione di un errore fonetico annotato | ≤ 5% |
| falso rifiuto di una pronuncia corretta | ≤ 10% |
| risultati `UNDECIDABLE` su audio valutabile | ≤ 20% |
| latenza finale p95 sull'hardware target | ≤ 1,5 s |
| memoria residente del worker fonetico | ≤ 2 GB |

Questi valori sono requisiti da verificare, non prestazioni dichiarate. Finché
non esiste il dataset learner annotato, la calibrazione resta marcata
`en-us-mvp-uncalibrated-v1` e non può essere presentata come validazione di
prodotto.

## 9.1 Boundary fra streaming e inferenza fonetica

Il WebSocket continua a produrre `stream.inference.partial` ogni 40 ms e
aggiorna il DTW rolling ogni 200 ms. Tali eventi contengono VAD, qualità e
somiglianza provvisoria, ma non un giudizio di pronuncia.

La sequenza finale è:

```text
frame 40 ms -> VAD/telemetria/DTW provvisorio
session.finish -> segmentazione e cleaning dell'utterance completa
               -> word identity gate
               -> una sola inferenza CTC fonetica
               -> forced alignment e scoring
               -> stream.inference.final
```

Non si chunka il modello fonetico nell'MVP: sulle parole brevi il contesto
completo vale più di una parziale riduzione della latenza. Un backend truly
streaming potrà essere aggiunto dietro il protocollo solo dopo misure
comparative.

## 10. Piano di migrazione

### Fase 1 — Contratti e osservabilità

- introdurre `PhonemeModel`, `ForcedAligner` e `PronunciationScorer`;
- aggiungere selezione del microfono e calibrazione automatica del rumore;
- mostrare distanze, confidenze e motivazioni nella demo;
- mantenere il DTW come baseline.

Criterio di completamento: ogni decisione è spiegabile e versionata.

### Fase 2 — Word identity gate

- integrare un riconoscitore a vocabolario chiuso;
- rifiutare parole diverse con confidenza calibrata;
- aggiungere test con tutte le coppie delle parole disponibili.

Criterio di completamento: dire una parola nota diversa dalla richiesta non può
produrre `CORRECT` oltre il limite di falsa accettazione stabilito.

### Fase 3 — Modello fonetico e allineamento

- integrare il modello fonetico dietro il protocollo;
- implementare allineamento alla pronuncia del lessico;
- salvare evidenza per fonema;
- restituire `UNDECIDABLE` quando l'allineamento è debole.

Criterio di completamento: il sistema localizza sostituzioni, cancellazioni e
inserzioni sul dataset annotato.

### Fase 4 — Calibrazione degli errori

- acquisire o integrare learner speech annotato;
- calibrare soglie per fonema e contesto;
- congelare un test set speaker-disjoint;
- produrre un report riproducibile.

Criterio di completamento: le metriche sul test set soddisfano i limiti di
accettazione definiti dal prodotto.

### Fase 5 — Feedback e validazione

- associare errori fonetici a suggerimenti pedagogici;
- far revisionare i messaggi a esperti;
- testare comprensibilità, equità e stabilità;
- definire consenso, retention e revisione umana.

Criterio di completamento: il feedback è utile, prudente e tracciabile.

## 11. Strategia di test

### Unit test

- regole del decision engine;
- gestione di inserzioni, cancellazioni e sostituzioni;
- più pronunce valide;
- versionamento e serializzazione;
- casi di qualità insufficiente.

### Integration test

- audio → modello → allineamento → punteggio → decisione;
- persistenza atomica di valutazione e fonemi;
- WebSocket con frame validi, persi e fuori ordine;
- compatibilità tra versioni degli artefatti.

### Black-box test

Per ogni parola:

- pronunce corrette mai viste;
- altre parole del vocabolario;
- errori fonetici annotati;
- silenzio, rumore, clipping e audio troncato;
- accenti e dispositivi differenti.

Un test che usa soltanto segnali sintetici non dimostra la qualità del
correttore. I test sintetici verificano il software; l'audio annotato verifica
il modello.

## 12. Decisioni e prerequisiti

Decisioni fissate per l'MVP:

1. lingua e locale: inglese `en-US`;
2. vocabolario: `bird`, `follow`, `forward`, `happy`, `learn`;
3. modello: `facebook/wav2vec2-xlsr-53-espeak-cv-ft`;
4. licenza del modello: Apache-2.0;
5. deployment: worker server-side, CPU come baseline;
6. inferenza: utterance completa a `session.finish`;
7. budget: p95 ≤ 1,5 s e RSS ≤ 2 GB sull'hardware target;
8. feedback non clinico, limitato a evidenza e suggerimenti revisionati.

Prerequisiti ancora bloccanti per dichiarare completate le fasi 3 e 4:

1. scegliere la popolazione e fascia d'età target;
2. ottenere learner speech con errori fonetici revisionati;
3. registrare anche l'hash locale dell'artefatto scaricato;
4. misurare i gate numerici della sezione 9;
5. revisionare lessico e messaggi con competenza fonetica.

## 13. Definizione di successo dell'MVP

L'MVP è completato quando:

- distingue la parola richiesta dalle altre parole del vocabolario;
- rileva almeno le classi di errore fonetico presenti nel dataset annotato;
- non restituisce `CORRECT` senza evidenza positiva su parola e fonemi;
- localizza l'errore con intervalli temporali;
- espone confidenza, motivazioni e versioni;
- supera un test speaker-disjoint con soglie definite prima della valutazione;
- conserva il comportamento `RETRY` e `UNDECIDABLE` nei casi non affidabili.
