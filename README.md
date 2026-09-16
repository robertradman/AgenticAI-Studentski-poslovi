# Agent za provjerenu pretragu studentskih poslova

Python agent koji pomaže studentima pronaći stvarne, trenutno aktivne studentske
poslove i prakse u Hrvatskoj. Ne vraća samo popis rezultata pretrage nego svaki
oglas prije prikaza i otvori i provjeri pojedinačno — ako ne može dokazati da je
oglas aktivan i da zadovoljava tražene uvjete, oglas se jednostavno ne prikazuje.
Cilj je bio napraviti agenta kojem korisnik može vjerovati kad kaže "nema
rezultata", umjesto agenta koji uvijek vrati nešto samo da bi izgledao koristan.

Projekt je namjerno ograničen na studentske poslove i prakse u Hrvatskoj i na
četiri unaprijed definirana portala. Ne pretražuje društvene mreže, Google
općenito niti bilo što izvan tih izvora.

## Sadržaj

- [Ideja iza projekta](#ideja-iza-projekta)
- [Mogućnosti](#mogućnosti)
- [Kako agent radi](#kako-agent-radi)
- [Alati koje agent koristi](#alati-koje-agent-koristi)
- [Heuristički način vs. LLM agentic način](#heuristički-način-vs-llm-agentic-način)
- [Pravila filtara i dokazivanje uvjeta](#pravila-filtara-i-dokazivanje-uvjeta)
- [Izvori i granice domene](#izvori-i-granice-domene)
- [Instalacija i postavljanje](#instalacija-i-postavljanje)
- [Pokretanje](#pokretanje)
- [Primjeri upita](#primjeri-upita)
- [Stanje, zapisi i izvještaji](#stanje-zapisi-i-izvještaji)
- [Testiranje](#testiranje)
- [Struktura projekta](#struktura-projekta)
- [Poznata ograničenja i mogući nastavak](#poznata-ograničenja-i-mogući-nastavak)

## Ideja iza projekta

Portali za studentske poslove (StudentPosao.hr, MojPosao, Posao.hr,
studentski-servis.hr) nemaju javni API, a oglasi na njima često imaju istekao
rok prijave, nejasan status ili se ne poklapaju s onim što korisnik traži
("isključivo vikendom", "bez iskustva", "najmanje 9 EUR/h" i slično). Umjesto
da agent samo parafrazira naslove oglasa, on:

1. pretražuje samo dopuštene izvore,
2. otvara pojedinačni URL svakog kandidata,
3. iz stvarnog sadržaja stranice izvlači dokaz (aktivnost, rok, satnicu,
   remote/vikend rad, traženo iskustvo),
4. i tek onda odlučuje ulazi li oglas u odgovor.

Ako podatak nije dokaziv iz sadržaja oglasa, agent ga ne izmišlja — radije
vrati manje rezultata ili nijedan, nego da pogodi.

## Mogućnosti

- Pretraga prema gradu, zanimanju i minimalnoj satnici.
- Provjera aktivnog oglasa, stvarnog datuma objave, satnice i vidljivog roka
  prijave — svaka tvrdnja u odgovoru ima izvor u dohvaćenom sadržaju stranice.
- Filtri za `remote`/online rad, isključivo vikendom, IT praksu, ljetnu praksu
  i rad bez prethodnog iskustva.
- Razlikovanje `više od 9 EUR/h` (strogo) od `najmanje 9 EUR/h` (uključno).
- Pamćenje preferencija između poziva (grad, ključne riječi) i prikaz samo
  novih oglasa od zadnjeg pregleda.
- Usporedba rokova prijave zadnja tri prikazana oglasa.
- Informativni izračun zarade s pragovima za status uzdržavanog člana i
  poreznu obvezu.
- Nacrt motivacijskog pisma za jedan točno određen, aktivan i provjeren oglas
  (pismo se ne šalje nigdje automatski, samo se ispiše kao prijedlog).

## Kako agent radi

```text
upit korisnika
  → tumačenje namjere i uvjeta (grad, satnica, iskustvo, vikend, remote, praksa...)
  → odabir rute (jednostavna pretraga / usporedba / izračun / stanje / status
     oglasa / motivacijsko pismo)
  → pretraga dopuštenih izvora
  → provjera pojedinačnog oglasa (fetch_listing)
  → primjena svih zadanih uvjeta nad dokazom iz oglasa, ne nad naslovom
  → rezultat s dokazima ili transparentna poruka da dokaza nema
```

`agent.py` je ulazna točka: tumači upit, gradi plan (`_build_plan`) i odabire
hoće li ga izvršiti deterministički (`_execute_plan`) ili predati Gemini
modelu kroz function calling (`_run_llm_agent`). `search_jobs.py` pretražuje
dopuštene izvore i radi jeftinu prvu filtraciju kandidata. `fetch_listing.py`
dohvaća pojedinačni oglas i iz njega izvlači sve podatke na kojima se temelji
odgovor. Ništa se ne prihvaća samo zato što se u naslovu pojavila tražena
riječ — mora postojati dokaz u tekstu oglasa.

Za aktivnost oglasa vrijedi ovo pravilo: koristi se ili stvarni sadržaj oglasa
(npr. "prijava do", tekst da oglas nije aktivan) ili činjenica da je oglas u
tom trenutku na službenom aktivnom popisu StudentPosao.hr, bez suprotnog
dokaza. Tehnički meta-podatak `unavailable_after`, koji neki portali koriste
za predmemoriranje stranice, smije samo potvrditi da je stranica dostupna —
**nikad** se ne prikazuje korisniku kao rok prijave, jer to nije.

## Alati koje agent koristi

| Alat | Što radi | Vanjski podaci? |
| --- | --- | --- |
| `search_jobs` | Pretražuje StudentPosao.hr preko službenog filtra (grad, satnica, online) te dodatno DuckDuckGo (`ddgs`) ograničeno na preostala tri portala. Svaki kandidat prije vraćanja prolazi kroz `fetch_listing` provjeru. | Da — pravi HTTP zahtjevi na portale. |
| `fetch_listing` | Otvara jedan URL oglasa i izvlači naslov, poslodavca, lokaciju, satnicu, rok prijave, aktivnost, remote/vikend rad i traženo iskustvo iz stvarnog HTML-a stranice. | Da |
| `estimate_earnings` | Računa procijenjenu zaradu za zadanu satnicu, sate tjedno i broj tjedana te uspoređuje s pragovima za status uzdržavanog člana (3600 EUR) i poreznu obvezu (12000 EUR). | Ne — čisti izračun. |
| `task_state` | Čita i sprema preferencije korisnika (grad, ključne riječi), evidentira viđene oglase (za "novi oglasi od zadnjeg puta") i drži zadnja tri prikazana oglasa za usporedbe. | Ne — lokalna datoteka. |
| `generate_application` | Prvo provjeri je li oglas aktivan (`fetch_listing`), pa tek onda Gemini modelu proslijedi samo podatke iz oglasa i profil kandidata da napiše nacrt motivacijskog pisma. Odbija raditi na neaktivnom ili neprovjerenom oglasu. | Da (neizravno, preko provjere oglasa) |

U LLM načinu rada, model kroz Gemini function calling sam odabire slijed ovih
alata i procjenjuje njihove rezultate prije sljedećeg koraka — to nije samo
jedan poziv modela koji "izmisli" JSON s parametrima.

## Heuristički način vs. LLM agentic način

| Način | Ponašanje |
| --- | --- |
| `heuristic` | Bez Gemini poziva. Radi determinističko tumačenje upita i pretragu preko istog `search_jobs`/`fetch_listing` lanca provjere. |
| `llm` | Zahtijeva `GEMINI_API_KEY`. Gemini kroz function calling bira alate, čita rezultate i odlučuje o sljedećem koraku (stvarna agentic petlja s više koraka). |
| `auto` | Zadano: koristi `llm` ako je ključ postavljen, inače pada natrag na `heuristic`. |
| `--offline` | Ne koristi ni Gemini ni mrežu; radi nad lokalnim `sample_results.json`, isključivo za demonstraciju/testove. |

Bitna napomena: **heuristički način nije agentic demonstracija projekta.**
Uveden je kao interna prečica za brzo testiranje logike filtriranja
(`_matches_criteria`, provjera satnice, iskustva i sl.) bez trošenja Gemini
poziva dok se ta logika mijenja, i za upite koji imaju jasne, doslovne uvjete
gdje bi parafraziranje modela moglo promašiti službeni filtar portala (npr.
"isključivo vikendom" ne smije postati obična pretraga za "vikend posao").
Sve što je stvarno višekoračno — usporedbe, rad sa spremljenim stanjem,
provjera statusa oglasa, izračun zarade, motivacijsko pismo — heuristički
način svjesno odbija izvršiti i traži pokretanje s `--mode llm`. Za
demonstraciju agentic ponašanja treba koristiti `llm` način s postavljenim
`GEMINI_API_KEY`.

Determinizam za jednostavne pretrage nije odstupanje od agentic pristupa nego
zaštita: garantira da model ne može "olabaviti" uvjet poput "bez iskustva" ili
"isključivo vikendom" tijekom parafraziranja upita. Kad je Gemini uključen,
može birati isključivo među pet dopuštenih alata iz tablice iznad.

### Postavljanje Gemini ključa

```powershell
$env:GEMINI_API_KEY="vaš_ključ"
python agent.py --mode llm "Usporedi tri najnovija ugostiteljska posla u Osijeku po satnici i radnim satima"
```

### Varijable okoline

| Varijabla | Zadano | Svrha |
| --- | --- | --- |
| `GEMINI_API_KEY` | nije postavljeno | Obvezan za LLM function calling i motivacijsko pismo. |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Model koji se koristi i za glavnu petlju i za motivacijsko pismo. |
| `GEMINI_MIN_CALL_INTERVAL_SECONDS` | `0` | Dodatni razmak između Gemini poziva ako to zahtijeva API plan. |
| `AGENT_MODE` | `auto` | Zadani način rada ako `--mode` nije naveden. |
| `AGENT_OFFLINE` | nije postavljeno | Postavite na `1` za lokalni offline rad bez mreže. |

Na Gemini odgovore `429` (RESOURCE_EXHAUSTED) i `503` (UNAVAILABLE) koristi se
ograničeni eksponencijalni ponovni pokušaj (`call_model_with_backoff`); nema
namjernog čekanja nakon uspješnog poziva.

## Pravila filtara i dokazivanje uvjeta

| Uvjet u upitu | Što agent prihvaća kao dokaz |
| --- | --- |
| `više od 9 EUR/h` | Satnica strogo veća od 9.00 EUR/h. |
| `najmanje 9 EUR/h`, `>= 9 EUR/h` | Satnica 9.00 EUR/h ili veća. |
| `bez prethodnog iskustva` | Izričit tekst u oglasu: "iskustvo nije uvjet", "bez prethodnog iskustva" i slične formulacije — ne prihvaća se šutnja o iskustvu kao dokaz. |
| `isključivo vikendom` | Dokaz rada samo subotom/nedjeljom; formulacije poput "ne radimo vikendom" izričito se odbijaju. |
| `online`, `remote`, `rad od kuće` | Službeni online filtar StudentPosao.hr i/ili izričit navod u tekstu oglasa. |
| `IT praksa` | Istodobno IT dokaz **i** dokaz prakse/internshipa/trainee programa — posao developera sam po sebi nije praksa. |
| `ljetna praksa` | Istodobno dokaz prakse i ljetnog programa. |
| `rok poslije 15. rujna` | Vidljiv rok prijave strogo poslije navedenog datuma. |
| `najnoviji oglasi` | Samo oglasi s čitljivim, nedvosmislenim datumom objave; sortiranje po tom datumu. |

Zato strogo filtriran upit može vratiti manje oglasa ili nijedan — to je
očekivano ponašanje, ne greška. Poruka "nema rezultata" znači da trenutačno
nije pronađen oglas koji je istodobno s dopuštenog izvora, aktivan i dokazivo
zadovoljava sve uvjete; ne znači da agent tvrdi kako takav oglas ne postoji
nigdje na internetu.

## Izvori i granice domene

Pretražuju se isključivo javno dostupni oglasi sa sljedeća četiri portala:

- [StudentPosao.hr](https://studentposao.hr/) — primarni izvor; koriste se
  njegovi službeni filtri za grad, satnicu i online rad.
- [MojPosao](https://www.moj-posao.net/)
- [Posao.hr](https://www.posao.hr/)
- [studentski-servis.hr](https://studentski-servis.hr/)

Uz primarni dohvat StudentPosao.hr, za svaki upit postoji jedna ograničena
dodatna pretraga preostala tri dopuštena izvora (preko `ddgs`, sa `site:`
filtrom na te domene). URL kandidata mora biti pojedinačni oglas na
dopuštenoj domeni (npr. `posao.hr/oglasi/...`, ne `posao.hr/gradovi/osijek/`)
— agent to provjerava prije nego uopće pokuša dohvatiti stranicu. Facebook i
svi drugi izvori su izvan opsega projekta.

## Instalacija i postavljanje

### Preduvjeti

- Python 3.11 ili noviji
- internetska veza za aktualne oglase (nije potrebna samo u `--offline` načinu)
- Gemini API ključ obvezan za višekoračni agentic tok: usporedbe, rad sa
  stanjem, status oglasa, izračun zarade i motivacijska pisma

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ako je aktivacija blokirana samo za trenutačnu PowerShell sesiju:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux i macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Projekt koristi tri vanjske biblioteke: `google-genai` (Gemini klijent),
`ddgs` (pretraga preostalih portala) i `requests` (dohvat pojedinačnih
oglasa).

Ako `.venv` navodi Python instalaciju koja više ne postoji, izbrišite samo
mapu `.venv` projekta i ponovite stvaranje virtualne okoline. Ne brišite
`data/` jer ona sadrži spremljeno stanje (preferencije, viđeni oglasi).

## Pokretanje

Osnovni oblik je:

```powershell
python agent.py "vaš upit"
```

### Jednostavne pretrage bez Gemini ključa

```powershell
python agent.py --mode heuristic "Pronađi studentske poslove u Osijeku koji plaćaju više od 9 EUR/h"
python agent.py --mode heuristic "Pronađi studentske poslove bez prethodnog iskustva"
python agent.py --mode heuristic "Pronađi studentske poslove koji se rade isključivo vikendom"
python agent.py --mode heuristic "Pronađi studentske poslove koji se mogu raditi online ili remote"
python agent.py --mode heuristic "Pronađi studentske oglase objavljene zadnja 3 dana za remote rad"
```

Heuristički način je namijenjen samo jasnim filtriranim pretragama (vidi
napomenu u prethodnom poglavlju) — ne pokušava oponašati planiranje,
usporedbu ni generiranje teksta bez modela.

### Višekoračni agentic upiti s Gemini ključem

```powershell
$env:GEMINI_API_KEY="vaš_ključ"
python agent.py --mode llm "Ima li trenutno otvorenih ljetnih praksi za informatiku u Zagrebu s rokom prijave poslije 15. rujna?"
python agent.py --mode llm "Usporedi 3 najnovija ugostiteljska posla u Osijeku po satnici i satima tjedno."
python agent.py --mode llm "Ako radim cijelu godinu 20 sati tjedno za 7 EUR/h, kolika je zarada i jesam li blizu pragova?"
python agent.py --mode llm "Daj kratak pregled 3 najbolje plaćena oglasa za šankera u Osijeku ovaj tjedan."
python agent.py --mode llm "Provjeri je li oglas za Junior Software Developer još uvijek aktivan."
python agent.py --mode llm "Zapamti da gledam samo IT poslove u Osijeku."
python agent.py --mode llm "Od zadnja tri oglasa koja sam vidio, koji ima najraniji rok prijave?"
```

CLI prihvaća cijeli preostali tekst upita kao jedan argument, pa unutarnji
navodnici oko naslova ne uzrokuju `unrecognized arguments`. Najjednostavnije
je ipak koristiti jedne vanjske navodnike oko cijelog upita, kao u primjerima
iznad.

### Motivacijsko pismo s Gemini ključem

```powershell
python agent.py --mode llm "Napiši mi motivacijsko pismo za posao Junior Software Developer. Profil: Student 3. godine prijediplomskog studija matematike i računarstva, imam iskustva s raznim programskim jezicima."
```

Gemini u alatskoj petlji najprije pretraži i provjeri oglas, a tek onda poziva
alat za generiranje pisma. Pismo se smije izraditi samo za aktivan provjeren
oglas i uneseni profil; ne šalje se automatski nikome.

### Offline demonstracija

```powershell
python agent.py --offline "Pronađi studentske poslove u Osijeku"
```

Offline način koristi `sample_results.json` (nekoliko ručno napisanih
primjera oglasa), bez mreže i bez stvarne provjere sadržaja. Namijenjen je
isključivo demonstraciji i testovima kad nema internetske veze ili API ključa.

## Primjeri upita

Ovo su konkretni upiti korišteni i za razvoj i za evaluaciju (`evaluate.py`),
pa dobro pokazuju raspon onoga što agent treba znati odraditi:

1. Pronađi studentske poslove u Osijeku koji plaćaju više od 8 EUR/h i ne
   traže iskustvo.
2. Jesu li otvorene ljetne IT prakse u Zagrebu s rokom prijave nakon 1. rujna?
3. Usporedi 3 najnovija ugostiteljska posla u Osijeku po satnici i satima
   tjedno.
4. Pronađi studentske poslove koji se rade isključivo vikendom.
5. Ako radim cijelu godinu 20 sati tjedno za 7 EUR/h, kolika je zarada i jesam
   li blizu pragova?
6. Pronađi studentske oglase objavljene zadnja 3 dana za remote rad.
7. Daj kratak pregled 3 najbolje plaćena oglasa za šankera u Osijeku ovaj
   tjedan.
8. Provjeri je li oglas "Zamjena nastavnika matematike u srednjoj školi" još
   aktivan.
9. Zapamti da gledam samo IT poslove u Osijeku.
10. Od zadnja 3 oglasa koja sam vidio, koji ima najraniji rok prijave?

Puni skup se pokreće preko `evaluate.py` (vidi [Testiranje](#testiranje)).

## Stanje, zapisi i izvještaji

```text
agent.py                 tumačenje upita, rute i konačni odgovor
search_jobs.py           pretraga portala, URL sigurnost i provjera kandidata
fetch_listing.py         detalji oglasa i dokazi za uvjete
generate_application.py  lokalni/Gemini nacrt motivacijskog pisma
estimate_earnings.py     informativni izračun zarade
task_state.py            preferencije, viđeni oglasi i zadnji prikazi
verify_live.py           read-only provjera stvarne pretrage
evaluate.py              pokretanje evaluacijskog skupa upita
sample_results.json      lokalni podaci za offline način
test_*.py                regresijski testovi
data/task_state.json     lokalno trajno stanje (nije u repozitoriju)
logs/execution_log.jsonl zapis izvršavanja — koji alat, s kojim argumentima, koliko koraka
reports/                 evaluacijski izvještaji (evaluate.py)
```

`data/task_state.json` pamti preferencije, kanonizirane URL-ove viđenih
oglasa i zadnjih do 20 prikaza (svaki zapis: naslov, URL, izvor, rok, satnica,
datum objave, je li aktivan/provjeren, kad je pogledan). URL se kanonizira
uklanjanjem UTM i klik-parametara (`utm_*`, `fbclid`, `gclid`, `mc_cid`,
`mc_eid`), tako da ista objava ne postane "nova" samo zbog drukčije poveznice.
Datoteka se zapisuje atomski (piše se u privremenu datoteku pa se
`os.replace`-a), a ako je oštećena, agent je namjerno **ne** zamjenjuje tihim
praznim stanjem — to bi izbrisalo povijest i vratilo stare oglase kao "nove".

`logs/execution_log.jsonl` je zapis izvršavanja bez internog rezoniranja
modela: svaki redak je jedan JSON događaj (`task_start`, `plan`, `tool_call`,
`tool_result`, `observation`, `recovery`, `review`, `task_end`...) s
vremenskom oznakom i ID-em sesije. Koristi se za provjeru koliko je koraka
agent napravio i koje je alate zvao za koji zadatak.

`evaluate.py` pokreće skup od deset upita iznad kroz `run_task` i sprema
izvještaj u `reports/evaluation_report.json`: za svaki upit bilježi ishod
(`completed`, `no_verified_results`, `needs_history`, `error`, ...), rutu,
broj koraka, koje je alate koristio i je li zadovoljio "ugovor" —
da je ishod uspješan/transparentan, da je korišten samo dopušteni skup alata i
da plan postoji.

## Testiranje

Pokretanje cijelog regresijskog skupa:

```powershell
python -m unittest -v
```

Skup ima dvije glavne datoteke i pokriva i rubne slučajeve:

- `test_search_jobs.py` — sigurnosne i filtarske provjere na razini pretrage:
  odbijanje URL-ova koji nisu pojedinačni oglasi, prepoznavanje neaktivnih
  oglasa, prihvaćanje budućih rokova, zahtjev za "studentski" dokazom osim na
  StudentPosao.hr domeni.
- `test_agent_features.py` — cijeli tok agenta: izračun zarade, spremanje i
  dedupliciranje stanja preko kanonskog URL-a, prepoznavanje kriterija iz
  prirodnog jezika (satnica, iskustvo, vikend, IT), ponašanje LLM petlje uz
  mockani Gemini klijent (da testovi ne troše stvarne API pozive).

Read-only provjera stvarne pretrage, bez izmjene spremljenog stanja:

```powershell
python verify_live.py "studentski posao" --city Osijek --min-wage 9 --require-results
```

Gemini evaluacijski skup (troši stvarne API pozive):

```powershell
python evaluate.py --mode llm
```

## Struktura projekta

```text
zavrsni/
├── agent.py                  glavna petlja, tumačenje upita, rute
├── search_jobs.py            pretraga i provjera kandidata
├── fetch_listing.py          dohvat i parsiranje pojedinačnog oglasa
├── generate_application.py   nacrt motivacijskog pisma
├── estimate_earnings.py      kalkulator zarade
├── task_state.py             spremanje preferencija i viđenih oglasa
├── verify_live.py            read-only CLI provjera pretrage
├── evaluate.py                pokretanje evaluacijskog skupa
├── sample_results.json       primjeri za offline način
├── requirements.txt
├── test_search_jobs.py
├── test_agent_features.py
├── data/
│   └── task_state.json       lokalno stanje (izostavljeno iz repozitorija)
├── logs/
│   └── execution_log.jsonl   zapis izvršavanja
└── reports/
    └── evaluation_report.json
```

## Poznata ograničenja i mogući nastavak

- Sadržaj oglasa i njihova dostupnost ovise o portalima u trenutku dohvaćanja
  — agent ne pamti povijesno stanje portala, samo ono što je upravo dohvatio.
- Broj kandidata po upitu je namjerno ograničen (do 4 sa StudentPosao.hr i 2 s
  preostalih izvora po pretrazi) radi zaštite izvora od prekomjernih zahtjeva,
  što znači da rijedak, ali stvaran rezultat teoretski može ostati neotkriven
  ako nije među prvih nekoliko kandidata.
- Izračun zarade je informativan, a pragovi nisu porezno ili pravno tumačenje.
- Motivacijsko pismo je nacrt koji korisnik mora pregledati i dopuniti prije
  slanja — ne uzima kontaktne podatke niti šalje ništa automatski.
- Parsiranje StudentPosao.hr kartica oslanja se na trenutnu HTML strukturu
  stranice (`job-card` klase, redoslijed atributa); promjena dizajna portala
  bi zahtijevala ažuriranje `_parse_studentposao_cards`.
- Prostor za nastavak: dodavanje još izvora unutar iste filozofije
  (provjeri-pa-vjeruj), keširanje rezultata pretrage (trenutačno se kešira
  samo pojedinačni odgovor stranice, na 5 minuta), i eventualno UI iznad
  postojeće CLI/agent logike.
