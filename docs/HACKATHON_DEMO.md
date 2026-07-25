# advX Hackathon Demo

## Cosa dimostra l'MVP

La UI presenta una frase completa e invia al backend:

- `expected_text`: la frase mostrata;
- `target_words`: la parola pedagogica da verificare;
- audio PCM mono 16 kHz in frame da 40 ms.

Il backend usa il keyword spotting soltanto come gate contro parole estranee.
La decisione di pronuncia viene invece dal modello fonetico CTC: allinea tutti
i fonemi della frase nota e valuta separatamente quelli della parola target.

Durante la registrazione la UI:

- reagisce al volume del microfono;
- avanza quando un segmento vocale viene acquisito dai frame da 40 ms;
- conclude automaticamente dopo il silenzio.

Al termine:

- `CORRECT` richiede evidenza positiva per tutti i fonemi target;
- `INCORRECT` indica una parola estranea oppure almeno un errore fonetico;
- `UNDECIDABLE` richiede un nuovo tentativo.

Senza modello fonetico disponibile il sistema restituisce `REVIEW_REQUIRED`:
non trasforma mai un semplice riconoscimento lessicale in pronuncia corretta.

## Avvio

```bash
cd /home/manuel/Scrivania/advX
make redis
make init
make dev
```

Aprire `http://localhost:8765` e fare un refresh completo (`Ctrl+Shift+R`) se
il browser aveva già caricato una versione precedente.

## Flusso demo

1. Premere **Start session**.
2. Premere il microfono una sola volta.
3. Leggere l'intera frase mostrata.
4. Attendere circa 800 ms in silenzio.
5. Mostrare l'evidenziazione progressiva e il risultato.

## Contratto WebSocket

```json
{
  "type": "session.start",
  "expected_text": "A happy bird can fly",
  "target_words": ["bird"],
  "locale": "en-US",
  "task": "sentence_pronunciation"
}
```

## Dopo l'hackathon

Evolvere il prototipo con:

1. inferenza CTC incrementale con cache dello stato acustico;
2. confini parola derivati dall'allineamento invece che dalle pause VAD;
3. score calibrati con SpeechOcean762 e dati advX;
4. validazione clinica con logopedisti.
