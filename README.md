# last-call

Webcam virtuale con effetti attivati dai gesti, da usare in qualsiasi app di videochiamata
(Meet, Zoom, Teams, FaceTime...). Gira tutto in locale: MediaPipe per mani, volto e corpo, OBS come webcam virtuale.

## Gesti ed effetti

| Gesto | Effetto | Tasto |
| --- | --- | --- |
| Indice contro indice e pollice contro pollice, a formare un ovale | **Squirtle**: acqua a schizzi verso la camera, gocce sull'obiettivo | `w` |
| Dito medio (anche con due mani) | Pioggia di 🖕 dal basso e dall'alto | `m` |
| Indici incrociati a croce | **Esorcismo**: occhi neri, testa che gira, latino maccheronico | `e` |
| Indice dritto verso l'alto, una mano sola in vista | **Mano di Dio**: scappellotto dal cielo e 🖕 divino | `d` |
| Braccia incrociate sul petto: la mano destra sulla spalla o sul braccio sinistro e viceversa (basta che si vedano le mani) | **Manifesto funebre** con la tua faccia | `f` |
| Pistola con le dita vista di lato (indice teso, pollice alzato), una mano sola | **WASTED**: spari che crepano lo schermo e schermata di GTA | `g` |
| Indice e medio uniti sul collo, a sentire il polso | **Elettrocardiogramma** che accelera e va piatto | `c` |
| Mani giunte | **Beatificazione**: aureola, raggi divini, "Dio 🐶", maiali in volo | `b` |
| Segno del time-out, mani a T: una dritta, l'altra di piatto sopra le sue dita | **La Morte** spunta dietro di te con falce e clessidra, ma cerca un altro | `o` |
| Mani nei capelli, tutte e due sopra la testa | **Sedia elettrica**: casco, scosse, raggi X con lo scheletro, fumo | `s` |
| Polsi uniti o incrociati e dita aperte verso l'alto, a fiamma | **Cremazione**: fiamme, cenere, urna in offerta | `r` |
| Mani in alto aperte, ai lati della testa | **Cecchini**: puntini laser su fronte e cuore, poi sparano: contraccolpo, sangue che ti segue e schizzi sull'obiettivo | `a` |
| Mano destra aperta sul cuore, dita verso la spalla sinistra | **Testamento** con lasciti a caso, firma e marca da bollo | `t` |
| Nessuno davanti alla camera per 3 secondi | **Lapide** col tuo nome; quando torni risorgi da zombie | `l` |

L'acqua e le 🖕 durano finché tieni il gesto e la lapide finché non torni; gli altri partono e vanno avanti
da soli per qualche secondo.

Nessun gesto usa quelli delle Reazioni di macOS (pollici su o giù, vittoria, corna a due mani). Due però ci
somigliano: l'ovale di Squirtle al cuore fatto con le mani, le due dita sul collo alla vittoria (vedi sotto).

## Uso

```bash
last-call --camera trust      # avvia, poi nell'app di videochiamata scegli "OBS Virtual Camera"
last-call --list-cameras      # elenca le fotocamere
last-call --nome Gigi         # nome su manifesto funebre, lapide e testamento (predefinito: il tuo)
last-call --preview --debug --no-virtualcam   # solo anteprima, con punti, gesto riconosciuto ed effetti attivi
```

Mentre gira, il terminale mostra un pannello con tutti i gesti: in giallo quello che sta vedendo (tienilo ancora
un attimo), in verde gli effetti attivi. I messaggi interni di MediaPipe vanno in `~/Library/Logs/last-call.log`.

Nell'anteprima i tasti della tabella lanciano gli effetti senza gesto, `q` esce. Altrimenti si esce con Ctrl+C.
L'anteprima mostra quello che vedono gli altri, con le scritte dritte; `--specchio` la mostra a specchio.

Le app di videochiamata mostrano a te la tua immagine a specchio, quindi lì vedi le scritte al contrario,
ma agli altri arrivano dritte. Su Zoom si può togliere in Impostazioni > Video > "Specchia il mio video".

## Installazione

Serve un Mac con [OBS Studio](https://obsproject.com) (fornisce la webcam virtuale) e [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Piergiuseppe/last-call.git
cd last-call
uv tool install --editable .
```

I modelli di MediaPipe si scaricano da soli in `src/last_call/models/` al primo avvio.

## Prima volta

1. Impostazioni di Sistema > Generali > Elementi di login ed estensioni > Estensioni fotocamera:
   attivare "OBS Virtual Camera".
2. Concedere la fotocamera all'app del terminale (macOS lo chiede al primo avvio).
3. Alcuni gesti somigliano a quelli delle Reazioni di macOS (cuore, pace): se compaiono anche gli effetti di Apple,
   disattivare Reazioni dal Centro di Controllo > Effetti video mentre il programma è in uso.

## Licenza

MIT, vedi [LICENSE](LICENSE).
