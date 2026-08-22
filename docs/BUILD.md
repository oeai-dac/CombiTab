# Bauen und Veröffentlichen

Für Mitwirkende. Wer CombiTab nur benutzen möchte, findet in
[INSTALLATION.md](INSTALLATION.md) das Passende.

## Voraussetzungen

Node.js 20 oder neuer. Sonst nichts — die Fachlogik kommt ohne Fremdbibliotheken
aus, und die Werkzeuge für den Paketbau lädt `electron-builder` selbst nach.

```bash
npm ci
```

> **Hinweis zu npm 12 und neuer:** npm blockiert Installations-Skripte von
> Abhängigkeiten inzwischen standardmäßig, und bei `npm ci` greift die
> `allowScripts`-Freigabe aus der `package.json` nicht zuverlässig. Electron
> käme dann ohne seine Programmdatei an. Deshalb holt das `postinstall`-Skript
> des Projekts (`scripts/ensure-electron.mjs`) den Download bei Bedarf nach.
> Sollte er einmal fehlschlagen: `npm run ensure-electron`.

## Alltag

```bash
npm run dev        # Entwicklungsserver im Browser
npm run build      # Typprüfung + Produktions-Build nach dist/
npm test           # gesamte Testsuite (316 Prüfungen, framework-frei)
npm run electron   # Build + Start im Desktop-Gehäuse
npm run smoke      # Selbsttest der Desktop-Fassung
npm run bench      # Performance-Benchmark
```

### Der Selbsttest

`npm run smoke` startet ein unsichtbares Electron-Fenster und prüft genau die
Eigenschaften, die beim Wechsel vom Browser ins Gehäuse brechen können. Die
wichtigste davon: Die App erzeugt ihre Rechen-Worker als **Modul-Worker**
(`new Worker(url, { type: "module" })`). Unter `file://` blockiert Chromium
deren Laden, womit Korrespondenzanalyse, Bootstrap und Score-Berechnung
ausfielen — ohne sichtbare Fehlermeldung.

Deshalb lädt der Hauptprozess die Oberfläche **nicht** über `file://`, sondern
über ein eigenes, als sicher registriertes Schema `app://` (siehe
`electron/main.js`). Das liefert zusätzlich einen stabilen Origin und damit
verlässliches IndexedDB (Autosave), localStorage (Theme, Sprache) und einen
sicheren Kontext für `CompressionStream` (Teilen-Link).

## Installationspakete bauen

```bash
npm run dist          # für das aktuelle Betriebssystem
npm run dist:linux    # AppImage, deb, rpm
npm run dist:win      # NSIS-Installer, portable .exe
npm run dist:mac      # dmg für Intel und Apple Silicon
```

Die Ergebnisse liegen in `release/`.

Ein Betriebssystem kann jeweils nur seine eigenen Pakete zuverlässig bauen.
Deshalb erledigt das die CI auf drei Runnern gleichzeitig; lokal ist der Bau
vor allem zum Prüfen gedacht.

### Bekannte Hürde unter Arch Linux

`deb` und `rpm` entstehen über `fpm`, dessen mitgeliefertes Ruby `libcrypt.so.1`
benötigt. Arch und Derivate liefern nur `libcrypt.so.2`. Abhilfe:

```bash
sudo pacman -S libxcrypt-compat
```

Das AppImage ist davon nicht betroffen. Auf dem Ubuntu-Runner der CI stellt sich
die Frage nicht.

## Zur Code-Signierung

Die Pakete werden **bewusst nicht** mit einem Zertifikat signiert — weder bei
Apple noch bei Microsoft. Die Folgen für Nutzer sind in
[INSTALLATION.md](INSTALLATION.md) beschrieben.

Eine Feinheit ist dennoch nötig, und sie lässt sich in der `package.json` nicht
kommentieren, weshalb sie hier steht:

```json
"mac": { "identity": "-", "hardenedRuntime": false }
```

- **`identity: "-"`** erzwingt eine **Ad-hoc-Signatur**. Das ist nicht dasselbe
  wie „nicht signieren": Wäre `identity` auf `null` gesetzt, unterbliebe die
  Signierung vollständig — und eine gänzlich unsignierte Anwendung **startet auf
  Apple Silicon überhaupt nicht**, macOS beendet sie sofort. Die Ad-hoc-Signatur
  macht die App lauffähig, hebt die Gatekeeper-Warnung aber nicht auf.
- **`hardenedRuntime: false`** ist bei Ad-hoc-Signatur zwingend. Andernfalls
  verwirft die Library-Validierung das vorsignierte Electron-Framework, weil es
  eine andere Team-ID trägt, und die App stürzt beim Start ab.

Sobald Zertifikate vorliegen, genügt es, die Secrets in der CI zu hinterlegen
(`CSC_LINK`, `CSC_KEY_PASSWORD` für Windows und macOS; `APPLE_ID`,
`APPLE_APP_SPECIFIC_PASSWORD`, `APPLE_TEAM_ID` für die Notarisierung) und in
`.github/workflows/release.yml` `CSC_IDENTITY_AUTO_DISCOVERY` zu entfernen sowie
`mac.identity` auf den Zertifikatsnamen zu setzen.

## Veröffentlichen

```bash
npm version 2.0.1        # hebt package.json und legt den Tag v2.0.1 an
git push --follow-tags
```

Der Tag löst `.github/workflows/release.yml` aus. Der Ablauf baut auf drei
Betriebssystemen, hängt alle Dateien an einen **Release-Entwurf** und stoppt
dort. Prüfen Sie die Dateien, ergänzen Sie den Begleittext und veröffentlichen
Sie das Release erst dann von Hand.

Ein Push auf `main` aktualisiert außerdem die Web-Fassung auf GitHub Pages
(`.github/workflows/pages.yml`).

## Schriften

Cormorant Garamond, JetBrains Mono und Outfit liegen als WOFF2 unter
`src/fonts/` im Repository und werden mitgeliefert — die App ruft beim Start
keinen Fremdserver auf. Sollen Schnitte oder Stärken geändert werden:

```bash
npm run vendor-fonts
```

Das Skript lädt sie neu und schreibt `src/fonts.css`. Alle drei Familien stehen
unter der SIL Open Font License 1.1; ihre Weitergabe im Bundle ist zulässig.
