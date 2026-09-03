# Domain sweep for Dew

Measured 2026-09-02, 17:55 to 20:40 UTC. 348 stem and TLD combinations were checked. Every
registration fact comes from the registry's own RDAP or port-43 WHOIS server, chosen from the IANA
bootstrap file or the IANA root zone database. No aggregator and no registrar search box was used to
decide availability. Prices come from the Porkbun public price API, registrar price pages, and
aftermarket landers that were rendered in headless Chrome.

This sweep replaces the narrow pass in `domain.md`, which looked at 18 TLDs. Two facts change the
recommendation. First, two exact-match `dew.` names left the registry at 20:34 UTC today, while this
sweep was running, and were still unregistered when it ended. Second, a dozen exact-match
`dew.<tld>` names are open at ordinary prices in TLDs the first pass never queried.

## Headline findings

1. `dew.tools` and `dew.software` **dropped during this sweep and are unregistered as of
   2026-09-02 20:39 UTC**. Both were held by the same registrant through Namecheap, both registered
   2025-06-17, both expired 2026-06-17, both entered `pendingDelete` at 2026-08-28 20:34 UTC. The
   predicted purge was 2026-09-02 20:34 UTC. A watcher polling the registry RDAP endpoint once a
   minute observed `dew.tools` flip from HTTP 200 to 404 at **20:34:03 UTC** and `dew.software` at
   **20:34:49 UTC**, matching the prediction to the minute. Both were still 404 five minutes later,
   so no drop-catcher had taken them at the last check. This is the moment to act, and it will not
   last.
2. `dew.ml`, the one exact "dew plus ml" address, is buyable, and it is expensive every year rather
   than once. The Mali registry quotes 250,000 F CFA a year. Gandi resells it at 569.39 USD for the
   first year and 598.45 USD a year after that.
3. Two names that RDAP reports as unregistered cannot actually be bought. `dew.site` is held back by
   the registry and `dew.tv` is a reserved string. An RDAP 404 alone does not mean registrable.
4. The word "dew" is already a product name in adjacent software. `trydew.app` serves "Dew, private
   period tracker for iPhone" and `usedew.app` serves "Dew, AI image generation studio". Neither
   name is available and both would be found by anyone searching for the project.

## Top ten

Scored on exact brand, total length, one-breath pronunciation, whether it reads as a developer tool,
whether the TLD forces HTTPS, registrar stability (on Cloudflare, standard tier not premium), and
confusion risk. Only names the owner can register are ranked. He is in the United States and is an
Indian national, so `.us` is usable and `.eu` is not. The renewal cutoff is 40 USD a year.

| Rank | Domain | First year / renewal (USD, Porkbun unless noted) | On Cloudflare | HTTPS forced by TLD | Why it is here |
| --- | --- | --- | --- | --- | --- |
| 1 | dew.tools | 9.78 / 29.35 at standard tier | yes | no | Bare brand plus a word that names the category. Reads as a developer tool with no explanation needed. Purged from the registry at 20:34:03 UTC today and unregistered at the last check. Try it at a registrar immediately; if it is still held in DropZone the attempt will fail and a backorder is the fallback. |
| 2 | dew.software | 15.96 / 33.47 at standard tier | yes | no | Purged at 20:34:49 UTC today, also unregistered at the last check. Exact brand and an accurate word for a framework. Twelve characters, and far less contested than a three-letter word in .tools, so the better odds of the two. |
| 3 | dew.science | 10.79 / 10.79 | yes | no | Cheapest exact match with a flat renewal. 10.79 every year, no premium tier, on Cloudflare. Reads as a research project, which fits a training framework better than it fits a product. |
| 4 | dewml.dev | 8.75 / 12.87 | yes | yes | The safe pick and the one already recommended. Matches the PyPI name dew-ml, browser-enforced HTTPS, cheapest renewal in the list. It adds no identity the package name does not already carry. |
| 5 | dew.works | 4.63 / 31.41 | yes | no | Exact brand and a natural phrase. Standard tier, on Cloudflare, renewal under the cutoff. The word says less about software than tools or software do. |
| 6 | dew.pro | 3.09 / 22.14 | yes | no | Exact brand, seven characters, the shortest open dew name that is not premium. Unrestricted since 2015, so the old credential rule does not apply. Reads corporate rather than technical. |
| 7 | dew.institute | 7.72 / 22.14 | yes | no | Exact brand at 22.14 renewal on Cloudflare. Thirteen characters and it names an organisation that does not exist, which is the same objection as dewlabs. |
| 8 | dew.page | 64.93 premium / 64.93 | yes | yes | Exact brand, browser-enforced HTTPS, on Cloudflare, and honest about being a docs page. Registry premium, so 64.93 to 76.70 recurs every year and the registry can move it. |
| 9 | dewkit.dev | 8.75 / 12.87 | yes | yes | Clean compound, one breath, standard .dev price, browser-enforced HTTPS. Sits behind dewml.dev only because kit says less than ml about what the project is. |
| 10 | dew.plus | 1,977 once, then 43.77 | yes | no | Exact brand available on the aftermarket for 1,977 buy now, the cheapest exact dew name with a real price attached. Renewal 43.77 is over the cutoff and plus says nothing about software. |

### The single best pick

`dew.tools`, and the window is open right now. It is the only name in the sweep that is the bare
brand and reads as a developer tool without a word of explanation, it is a standard-tier name rather
than a premium one, and Cloudflare Registrar carries `.tools`. It left the registry at 20:34:03 UTC
today and was still unregistered at 20:39:56 UTC.

What to do, in this order. Try to register `dew.tools` at any registrar that carries `.tools` right
now. If it fails, the name is still held in Identity Digital's DropZone, a separate EPP system with a
daily one-hour release window where registrars apply first-come-first-served; in that case place a
backorder with a drop-catch service, because a three-letter English word in `.tools` is exactly what
those services target. Cloudflare does not offer backorders, so a catch elsewhere would have to be
transferred to Cloudflare afterwards. Note that RDAP returning 404 does not distinguish "in DropZone"
from "openly registrable", which is why the answer is to try the checkout rather than to keep
watching RDAP.

If he wants the same idea with a real chance of success, `dew.software` is the better bet. Identical
drop timing, identical registry mechanics, but a twelve-character name in `.software` attracts far
less drop-catch attention than `dew.tools` will.

`dewml.dev` remains the answer if he wants to stop thinking about this today. It costs 8.75 USD for
the first year and 12.87 USD a year after, `.dev` is HSTS-preloaded so browsers refuse plain HTTP,
Cloudflare carries it, and it matches the PyPI name. The objection to it is unchanged: it
adds no identity that `pip install dew-ml` does not already carry.

The middle path is `dew.science` at 10.79 USD a year flat. It is the cheapest exact-match `dew.`
name with no premium tier and no restriction, and for a research framework the word is not wrong.

## dew.ml

This is the one exact "dew plus ml" address, so its real price matters.

| Vendor | Carries .ml | Standard first year / renewal | dew.ml first year | dew.ml renewal | Source |
| --- | --- | --- | --- | --- | --- |
| Registry direct (AGETIC, point.ml) | yes | 8,000 / 10,000 XOF, about 14.12 / 17.65 USD ex-VAT | 250,000 F CFA, about 441.26 USD ex-VAT, about 520.69 USD with Mali's 18 percent VAT | 250,000 F CFA a year | https://point.ml/ order flow |
| Gandi | yes, accredited since September 2024 | 25.54 / 55.98 USD | 569.39 USD | 598.45 USD a year | https://shop.gandi.net/en/domain/suggest?search=dew.ml |
| 101domain | yes, accredited | 51.99 / 79.99 USD | contact only, no published number | contact only | https://www.101domain.com/ml.htm |
| Dynadot | yes | 10.80 / 14.40 USD | unknown, search page behind a Turnstile challenge | unknown | https://www.dynadot.com/domain/prices |
| EuroDNS | yes | from 62.00 EUR | unknown | unknown | https://www.eurodns.com/domain-extensions/ml-domain-registration |
| Namecheap | no | not applicable | not offered, page says "Unsupported TLD" | not applicable | Namecheap search result |
| Porkbun | no | not applicable | not offered | not applicable | Porkbun price API has no `ml` key in 907 TLDs |

What the registry itself says. `https://rdap.nic.ml/domain/dew.ml` returns HTTP 200 with the notice
title "Premium String - Not Registered, Available as a Premium", the description "This domain has
not been registered. It is available as a premium name.", and a variant relation of
`RESTRICTED_REGISTRATION`. Port 43 to `whois.nic.ml` returns "Domain Status: Premium String - Not
Registered, Available as a Premium". Neither response carries a price.

Caveats that decide it.

- The premium is a recurring annual rate, not a one-time fee. The registry portal quotes 250,000
  F CFA for one year and 500,000 F CFA for two. Gandi's renewal is premium-priced too.
- No contract caps the renewal. Gandi's .ML Special Conditions v1.1, section ML.5, say premium names
  "are subject to specific prices as published on Our website during Your order".
- `.ml` is not HSTS-preloaded. There is no TLD-level `ml` entry in Chromium's preload list, which was
  verified against the same parse that finds `dev`, `app`, `page` and `day`.
- Cloudflare Registrar does not carry `.ml`, so its at-cost premium pricing does not apply.
- DNSSEC is not available for `.ml` per 101domain's registry information page.
- The registry is a Mali government agency (AGETIC). During this check the registration URL that
  IANA publishes, `http://www.nic.ml`, refused connections on ports 80 and 443, and all three of the
  registry's own policy PDFs returned 404. EuroDNS states the Malian state may take back any `.ml`
  domain on 15 days notice without compensation, though that is a registrar's restatement and the
  registry charter could not be retrieved to confirm it.
- Registration is open to anyone worldwide, so there is no eligibility problem.
- The Freenom-era abuse reputation was real. Spamhaus wrote in April 2023 that the Freenom ccTLDs,
  naming Mali, "have been a mainstay in virtually every statistic created around internet
  abuse-related domain registrations". Freenom's contract ended 2023-07-17 and registration is no
  longer free. `.ml` does not appear in Spamhaus's current worst-ten ccTLD lists for phishing,
  malware or botnet.

Verdict. `dew.ml` fails the 40 USD renewal bar by a factor of fifteen, and it fails it every year.
It is the exact name, but at about 600 USD a year with no price cap, no DNSSEC, no HSTS and no
Cloudflare support, it is the worst value in the sweep. Skip it.

## dew.tools and dew.software

Both names, queried at 2026-09-02 18:14 UTC:

| Field | dew.tools | dew.software |
| --- | --- | --- |
| RDAP status | `client transfer prohibited`, `pending delete` | `client transfer prohibited`, `pending delete` |
| Registration | 2025-06-17T20:28:53Z | 2025-06-17T20:28:53Z |
| Expiration | 2026-06-17T20:28:53Z | 2026-06-17T20:28:53Z |
| Last changed | 2026-08-28T20:34:16Z | 2026-08-28T20:34:21Z |
| Registrar | NameCheap, Inc. | NameCheap, Inc. |
| DNS | none | none |
| Porkbun price if it drops at standard tier | 9.78 first year, 29.35 renewal | 15.96 first year, 33.47 renewal |
| On Cloudflare | yes | yes |

Note that `whois.nic.tools`, `whois.donuts.co`, `whois.identitydigital.services` and
`whois.afilias.net` all answer "TLD is not supported" for these names. The IANA delegation record for
`.tools` lists no WHOIS server, only the RDAP base. Identity Digital serves RDAP only for these
TLDs, so RDAP is the sole authoritative source and it does not expose premium or reserved status.
The registry operator of record is Binky Moon, LLC c/o Identity Digital Inc.

### The drop timeline

The lifecycle numbers, from ICANN rather than from memory:

- Auto-renew grace period: 0 to 45 days, set by the registrar, not the registry. ICANN's
  "Life Cycle of a Typical gTLD Domain Name" infographic,
  `https://www.icann.org/en/registrars/gtld-lifecycle.jpg`.
- Redemption Grace Period: 30 days. ICANN Expired Registration Recovery Policy, section 3.1: "all
  gTLD registries must offer a Redemption Grace Period ("RGP") of 30 days immediately following the
  deletion of a registration".
- pendingDelete: 5 days. ICANN EPP status codes page, redemptionPeriod entry: "Your domain will be
  held in this status for 30 days. After five calendar days following the end of the
  redemptionPeriod, your domain is purged from the registry database and becomes available for
  registration."

The arithmetic for these two names. Expiry 2026-06-17T20:28:53Z. RDAP last changed
2026-08-28T20:34:16Z, which is the transition into pendingDelete. That is 72 days after expiry.
Subtracting the 30-day RGP puts the registrar's deletion at 2026-07-29, so Namecheap held the name
42 days past expiry, inside the 0 to 45 day window. 42 plus 30 plus 5 is 77 days, which sits inside
ICANN's 80-day maximum and inside Namecheap's own published band of about 70 to 80 days. Nothing
unusual happened: no restore, no registrar auction, no registry hold.

Purge instant: 2026-08-28T20:34:16Z plus 120 hours is 2026-09-02T20:34:16Z. Namecheap's own wording
("on the 6th day the domain should be released") would have put it on 2026-09-03, so the 5x24h
reading and the calendar reading straddled two days.

The 5x24h reading was right. A watcher polling both registry RDAP endpoints once a minute from
18:14 UTC recorded HTTP 200 for both names continuously, then:

```
2026-09-02T20:33:03Z dew.tools 200
2026-09-02T20:34:03Z dew.tools 404
2026-09-02T20:34:23Z dew.software 200
2026-09-02T20:34:49Z dew.software 404
```

Both then stayed 404 through 20:39:56 UTC, checked every 25 seconds, so neither had been re-registered
at the last observation. The predicted instant and the observed instant agree to within a minute.

### Why he cannot just race a registration form

This is the part that decides whether `dew.tools` is realistic. Purged Identity Digital names do not
go straight into open registration. They go to DropZone first. From Identity Digital's RSEP request
2023-002 filed with ICANN: "Dropzone is a discrete EPP server system, separated from the main
registry instance that will manage, on a daily basis, the synchronized release of expiring domain
names ... Applications are made on a first-come first-serve basis and awarded at the end of the
Dropzone period to successful applicants ... Where there are no successful applications for an
expiring domain, the domain will be released for open registrations in the primary registry instance
at the end of the Dropzone awarding process."

Their registrar page describes it as "a daily one-hour release window". Their Registry Terms and
Conditions, section 7.2, say the primary EPP service "does not support DropCatching or Bulk Expiring
Domain Traffic". So the only way to get `dew.tools` is through a registrar or drop-catch service that
applies in the DropZone window. A three-letter English word in `.tools` is exactly the kind of name
those services target. Cloudflare Registrar does not do backorders, so this has to be done somewhere
else and transferred later if he wants it on Cloudflare.

Two further caveats. The registry may re-rate any unregistered name, so the standard price above is
only knowable at the moment it drops (Registry Terms and Conditions, section 5.2, 30 days notice to
change a Name Rating). And DropZone may charge a per-TLD application fee on top of the registration
price, per the same RSEP filing. Identity Digital publishes a daily curated drop list at
`https://files.identity.digital/droplist-identity-digital/domain_drop_finder.csv`; on 2026-09-02 it
held 1000 rows with six `.tools` names and no `dew` entry, but that list is a marketing subset, so
absence from it does not mean the name skips DropZone.

## Names that RDAP says are free but are not

| Domain | RDAP | Registry WHOIS | Meaning |
| --- | --- | --- | --- |
| dew.site | 404 | "This name is not available for registration: Domains in the list are kept aside due to potential value, eventual goal would be to add caught domains to the premium list in a price change" | Held by the registry, not registrable |
| dew.tv | 404 | "Reserved Domain Name" | Reserved string, not registrable |
| dew.ml | 200 with premium notice | "Premium String - Not Registered, Available as a Premium" | Registrable at about 600 USD a year |
| dew.me | not applicable | "This premium domain is available for purchase. To register the domain, please contact premium@identity.digital." then "Domain not found." | Registrable only through Identity Digital's premium desk, price not published |

Where a registry WHOIS server answered, it confirmed availability positively for `dew.tech`,
`dew.space`, `dew.online`, `dew.click`, `dew.link`, `dew.in` and `dewml.cloud` with the wording
"Domain <name> is available for registration". For the Identity Digital and Google Registry TLDs no
WHOIS answer is available, so premium or reserved status there is unverified and has to be confirmed
at a registrar's checkout.

## Confusion risk

The parent word is crowded. These are live sites found in this sweep, not guesses.

| Domain | What is actually served |
| --- | --- |
| trydew.app | "Dew, private period tracker for iPhone" |
| usedew.app | "Dew, AI image generation studio" |
| dew.dev | "DewDev" |
| dew.sh | "DEW.SH, a Dub custom domain" |
| dew.gg | "Dew, #1 NFT aggregator on Polygon" |
| dewy.io | "10x your patient follow-up with AI, Dewy" |
| dewlabs.io | "Dew Labs, dew point based humidity control" |
| dew.studio | "Home, Dew" |
| dew.la | "D.E.W. L.A., operations, marketing and web development" |
| dewpoint.com | "Managed IT, cybersecurity and technology consulting, Dewpoint" |
| dewpoint.in | "Indoor air quality consultants and smart HVAC solutions" |
| dewlab.org | "DEW Lab, Teachers College" |
| dew.academy | "dew involvement GmbH" |

Two of these are software products literally branded "Dew", one of them in AI. That is an argument
for a TLD that carries the category, such as `.tools` or `.software`, over one that does not.

## Aftermarket prices found

Buy-now prices read from rendered landers. No offers were made.

| Domain | Price | Channel |
| --- | --- | --- |
| dew.expert | 750 USD | Afternic |
| dew.plus | 1,977 USD (payment plans at 494.25 and 247.12 shown) | Spaceship |
| dew.design | 2,500 USD | Afternic |
| dew.one | 5,950 USD | Afternic |
| dew.so | 5,990 USD | Spaceship |
| dew.io | 74,995 USD, or lease 1,625 a month | GoDaddy |
| dew.ai | offers from 99,500 USD | Spaceship |
| dew.org | 135,000 USD, or lease 4,744 a month | GoDaddy |
| dew.xyz | 198,795 USD | Spaceship |
| dewhq.com | 2,888 USD | Afternic |
| dewlab.com | 395.63 USD | HugeDomains |
| dewlabs.com | 100 USD | HugeDomains |
| dewdrop.app | 999 USD | Spaceship |
| dewy.ai | 167,881 USD | Spaceship |
| dew.eu | listed for sale, no price shown | owner's own page |

## HTTPS enforced by the TLD

Only five TLDs in the whole grid have a TLD-level `force-https` entry with `include_subdomains` true
in Chromium's preload list: `app`, `day`, `dev`, `new` and `page`. Exact entry for `.dev`:

```
{ "name": "dev", "policy": "public-suffix", "mode": "force-https", "include_subdomains": true }
```

Source: `https://raw.githubusercontent.com/chromium/chromium/main/net/http/transport_security_state_static.json`.
Every other candidate TLD needs HSTS set as a response header on the host. `.how` is a Google
Registry TLD but is not preloaded.

## Restrictions that apply

- `.new` (Google Registry): anyone may register, but the name must "resolve to the action within 100
  days of registration" and be an action-generation or content-creation flow, and Google Registry may
  verify compliance and suspend or delete a name that does not comply. Source `https://get.new/`. At
  412.46 USD a year it is out on price as well.
- `.pro`: unrestricted since 2015-11-16. The original professional-credential requirement is gone.
  Registry is Identity Digital. Source: OpenSRS .PRO domain policies, which states "As of November
  2015, the .PRO registry no longer requires any additional data requirements".
- `.eu`: not usable. Regulation (EU) 2019/517 Article 3 limits registration to a Union citizen
  regardless of residence, a non-Union citizen who is resident of a Member State, an undertaking
  established in the Union, or an organisation established in the Union. A United States resident who
  is an Indian national is none of those four. Source
  `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019R0517`.
- `.us`: usable. The usTLD Nexus Requirements Policy, Nexus Category 1, admits a natural person
  "(iii) whose primary place of domicile is in the United States of America or any of its
  possessions", so citizenship is not required and US domicile is enough. Two practical catches from
  the same document: the registrant must certify the Nexus category at registration, and must certify
  "that the listed name servers are located within the United States". Failure to satisfy Nexus puts
  the name on a 30-day hold and then cancels it with no refund. Source
  `https://www.about.us/documents/policies/usTLD_Nexus_Requirements_Policy.pdf`.
- `.ml`: open to anyone, one to ten year terms, no DNSSEC, no HSTS, government-run registry.
- `.dev`, `.app`, `.page`, `.day`: HSTS-preloaded, so a valid certificate is required from day one.
- Premium names on Google Registry TLDs carry a recurring premium fee that the registry may re-rate,
  per the `.dev` and `.page` pricing policies.

### Every ccTLD in the grid, eligibility and price

Prices in USD. Where Porkbun does not carry the TLD (`so`, `is`, `ax`, `ee`, `ml`) the figures come
from 101domain or Gandi. Eligibility was read from the registry rules a registrar restates, or from
the registry itself where it publishes them.

| ccTLD | Country and registry | Open to a US resident who is an Indian national | First year | Renewal | Term (years) | On Cloudflare | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| .so | Somalia | no | not published | 183.98 | 1 to 10 | no | Gandi's registry rules state: '.SO domain names are open to persons having a bona fide connection to Somalia, which includes but are not limited to: institutions and organizations |
| .is | Iceland | yes | 125.99 | 164.99 | 1 to 1 | no | Open to anyone worldwide. 101domain FAQ for .is: no special registration requirements; note only 'If you are an Icelandic citizen, please provide your Kennitala number.' . IS-NIC i |
| .in | India | yes | 7.83 | 7.83 | 1 to 10 | no | Open to anyone. Gandi: '.in is open to anyone.' 1-10 year terms. . As an Indian national the registrant also qualifies under NIXI's connection-to-India criterion, but it is not req |
| .us | United States | yes | 4.43 | 7 | 1 to 10 | yes | usTLD Nexus Requirements Policy, Category 1: a natural person 'who is a United States citizen, (ii) a permanent resident of the United States of America or any of its possessions |
| .me | Montenegro | yes | 2.73 | 17.27 | 1 to 10 | no | Gandi: '.ME domain names are open to everyone.' . Identity Digital is the technical operator (Cloudflare TLD policy list) |
| .cc | Cocos (Keeling) Islands (Australia) | yes | 3.4 | 8.55 | 1 to 10 | yes | Open to anyone. Gandi: '.cc domain names are open to everyone.' . Verisign-operated, no nexus rules |
| .tv | Tuvalu | yes | 26.26 | 26.26 | 1 to 10 | yes | Open to anyone. Gandi: '.tv domain names are open to everyone.' |
| .to | Tonga | yes | 51.8 | 51.8 | 2 to 10 | no | Gandi: '.TO domain names are open to everyone.' . Registration periods: 2, 3, 5 or 10 years |
| .la | Laos | yes | 27.34 | 27.94 | 1 to 10 | no | Open to anyone. Gandi: '.LA domain names are open to everyone. Gandi does not manage .LA Premium domain names.' . Explicit premium tier at registry level |
| .gg | Guernsey (Channel Islands) | yes | 51.8 | 51.8 | 1 to 1 | no | Open to anyone. Gandi: '.GG domain names are open to everybody.' 1-year registration terms only |
| .fm | Micronesia (Federated States of) | yes | 87.85 | 87.85 | 1 to 10 | yes | Open to anyone. Gandi: '.fm domain names are open to everyone.' |
| .am | Armenia | yes | 36.35 | 36.35 | 1 to 5 | no | Open to anyone. Gandi: '.AM domain names are open to everyone'. Registration period 1 to 5 years |
| .ax | Åland Islands (Finland) | yes | 131.49 | 164.99 | 1 to 5 | no | Gandi's rules section states '.AX domain names are open to everyone' with 1-year registration periods , contradicting the folklore that .ax requires an Åland connection. However 10 |
| .ee | Estonia | yes | 54.49 | 67.99 | 1 to 10 | no | Open to everyone; the historical local-contact rule is gone. Gandi: '.EE domains are open to everyone.' but registration requires 'individuals: your ID number and birthday, legal p |
| .ws | Samoa | yes | 22.97 | 22.97 | 1 to 10 | no | Open to anyone. Gandi: '.WS domain names are open to everyone.' |
| .eu | European Union | no | 5.88 | 5.88 | 1 to 10 | no | As of 2 August 2021, .eu registration is limited to: (1) a citizen of an EU Member State, Iceland, Liechtenstein or Norway regardless of residence; (2) a natural person resident in |
| .io | British Indian Ocean Territory (Chagos Archipelago) | yes | 28.12 | 51.8 | 1 to 10 | yes | Open to anyone. Registry's own site: 'Unlike some country code domains, you are not required to live in British Indian Ocean Territory to register .io domains' and 'Anyone, anywher |
| .sh | Saint Helena | yes | 31.2 | 46.65 | 1 to 10 | no | Open to anyone. Gandi: '.sh domain names are open to everyone.' . Note: .sh is also the ccTLD for Saint Helena but carries no residency requirement |
| .ai | Anguilla (UK overseas territory) | yes | 82.7 | 82.7 | 2 to 10 | yes | Open to anyone worldwide. Gandi: '.ai domains are open to everyone.' Registration period 2 to 10 years |
| .ml | Mali | yes | not published | premium only | 1 to unknown | no | Open to anyone. 101domain: 'There are no special requirements to register a .ML domain, which means anyone in the world can buy it.' Gandi: '.ml domains are open to everyone.' (Sou |

Reading this table. Only eight of these are both usable and under the 40 USD renewal cutoff: `.in`,
`.us`, `.me`, `.cc`, `.tv`, `.la`, `.am` and `.ws`. Of those, `dew.tv` is a reserved string, and
`dew.in`, `dew.us`, `dew.cc` and `dew.la` are already registered, `dew.me` is registry premium with
no published price, and `dew.am` and `dew.ws` are open. Two corrections to common belief came out of
this: `.ax` is open to everyone rather than restricted to an Aland connection, and Estonia's
historical local-contact rule for `.ee` is gone. Both are still priced far above the cutoff. `.so` is
the one ccTLD here the owner genuinely cannot use, because the registry requires a bona fide
connection to Somalia, which also settles `dew.so` and its 5,990 USD asking price.

## Method

- RDAP: `GET <server>/domain/<name>` with `Accept: application/rdap+json`. Servers taken from
  `https://data.iana.org/rdap/dns.json`, publication 2026-07-23T02:00:03Z. 404 means unregistered,
  200 means registered, except at `rdap.nic.ml` where a 200 with a premium notice and no registration
  event means unregistered premium.
- WHOIS: raw port 43 to the server named in the IANA root zone database, for the TLDs the bootstrap
  file does not cover (`io`, `sh`, `so`, `us`, `me`, `la`, `gg`, `am`, `ax`, `ee`, `ws`, `eu`).
- Queries were serialised per server, because 24 concurrent WHOIS queries produced empty responses
  that a naive parser would read as "not found". Two controls were run per WHOIS registry, a
  known-registered name and a nonsense name, to confirm the parser distinguishes them.
- One parser bug was found and fixed: the `.so` registry echoes `Domain Name: <query>` before
  "The queried object does not exist", which a keyword parser reads as a registration. All twenty
  `.so` rows were re-parsed. This is the same class of error that produced the wrong `dew.sh`
  result in the earlier pass.
- Live sites: DNS lookup, then HTTPS and HTTP GET, reading the title and scanning the body for
  sale wording and prices. GoDaddy landers return 403 to non-interactive clients, so Afternic listing
  pages were rendered with headless Chrome instead.
- Prices: `https://api.porkbun.com/api/json/v3/pricing/get` for standard per-TLD prices, all 68
  candidate TLDs present in the 907 the API returns. Second source: Dynadot's public list at
  `https://www.dynadot.com/domain/tlds`, whose numbers are in the page's own embedded SSR payload.
  Dynadot agrees with Porkbun within about a dollar on every TLD in the top ten, for example tools
  9.85 first year and 30.18 renewal against Porkbun's 9.78 and 29.35, and science 10.90 flat against
  10.79 flat. Porkbun's search page blocks non-interactive clients ("Hardcore hacker detected"), so
  per-name premium flags could not be read there in this session.
- Cloudflare support: `https://www.cloudflare.com/tld-policies/`, which Cloudflare's own Registrar
  API reference names as the list of supported TLDs.
- HSTS: the preload list was parsed twice independently, once by this agent and once by a second
  agent that decoded the full 10,521,810-byte file and enumerated all 94,644 entries with `jq`. Both
  passes returned the same five TLDs.

### Where this sweep is uncertain

- Per-name premium status is unverified for the Identity Digital and Google Registry TLDs, because
  their WHOIS is retired and their RDAP does not carry the flag. That covers `tools`, `software`,
  `digital`, `institute`, `engineering`, `fyi`, `run`, `training`, `center`, `guru`, `network`,
  `place`, `pro`, `team`, `works`, `dev`, `app`, `page`, `day` and `how`. Confirm the tier at
  checkout before buying.
- `dew.me` is registrable but its price is not published anywhere. It requires an email to
  `premium@identity.digital`.
- Expiry dates for `dew.is`, `dew.gg` and `dew.eu` were not returned in a parseable field by those
  registries.
- The Cloudflare TLD Policies page renders its table client-side and does not return the same markup
  on every fetch. Two independent extractions agreed on every TLD in the top ten, and both agree that
  `sh`, `ml` and `build` are absent, but they disagreed on `study`, `one`, `art`, `link`, `click`
  and `me`. Treat the Cloudflare column for those six as unconfirmed and check the dashboard. A
  third extraction, an exact-match scan restricted to the ccTLDs, found only `us`, `cc`, `tv`, `fm`,
  `io` and `ai` present, which is the figure used in the ccTLD table above.
- The drop watcher used for `dew.tools` and `dew.software` is still running and logging to
  `/tmp/dewsweep/drop.log`, one line per name per minute in the form `<timestamp> <name> <http
  status>`. A `404` in that file is the moment the registry purged the name. As of 2026-09-02
  20:08 UTC every line is `200`. Note that `/tmp` was cleared once during this session, so the log
  may not survive.

## Appendix A: every TLD for the `dew` stem

| Domain | Registered | Evidence | Registrar | Expires | Live site | Sale or note | Porkbun first year / renewal | Cloudflare | HTTPS forced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dew.com | yes | RDAP 200 | GoDaddy.com, LLC | 2029-05-02 | HTTP 200, no title | none | 11.08 / 11.08 | yes | no |
| dew.net | yes | RDAP or registry WHOIS 200 (17:00-17:40 UTC pass, unchanged) | Net-Chinese Co., Ltd. | 2026-11-25 | no DNS | dormant | 12.52 / 12.52 | yes | no |
| dew.org | yes | RDAP or registry WHOIS 200 (17:00-17:40 UTC pass, unchanged) | GoDaddy.com, LLC | 2027-03-28 | Afternic lander | for sale 135,000, lease 4,744/mo | 7.98 / 11.84 | yes | no |
| dew.dev | yes | RDAP or registry WHOIS 200 (17:00-17:40 UTC pass, unchanged) | Dynadot LLC | 2027-07-17 | HTTPS 200, title "DewDev", Cloudflare NS | in use, not for sale | 8.75 / 12.87 | yes | yes |
| dew.app | yes | RDAP or registry WHOIS 200 (17:00-17:40 UTC pass, unchanged) | Namecamp Limited (Yay.com) | 2027-05-08 | registrar holding page | not for sale | 8.75 / 14.93 | yes | yes |
| dew.page | no | RDAP 404 | none | none | none | premium at registrars | 10.81 / 10.81 | yes | yes |
| dew.day | no | RDAP 404 | none | none | none | premium at registrars | 10.81 / 10.81 | yes | yes |
| dew.new | no | RDAP 404 | none | none | none | none | 412.46 / 412.46 | yes | yes |
| dew.ai | yes | RDAP or registry WHOIS 200 (17:00-17:40 UTC pass, unchanged) | NameCheap, Inc. | 2027-04-11 | title "dew.ai for sale, Spaceship.com" | offers from 99,500 (Spaceship) | 82.70 / 82.70 | yes | no |
| dew.io | yes | RDAP or registry WHOIS 200 (17:00-17:40 UTC pass, unchanged) | GoDaddy.com, LLC | 2026-11-18 | redirects to forsale.godaddy.com/forsale/dew.io | for sale 74,995, lease 1,625/mo | 28.12 / 51.80 | yes | no |
| dew.sh | yes | RDAP or registry WHOIS 200 (17:00-17:40 UTC pass, unchanged) | NameCheap, Inc. | 2027-09-25 | title "DEW.SH - A Dub Custom Domain" | in use as Dub short-link domain | 31.20 / 46.65 | no | no |
| dew.ml | no | registry WHOIS: "Premium String - Not Registered, Available as a Premium" | none | none | none | registry premium | not carried | no | no |
| dew.run | yes | RDAP or registry WHOIS 200 (17:00-17:40 UTC pass, unchanged) | Alibaba Cloud (HiChina) | 2027-04-13 | no DNS | dormant | 4.12 / 22.14 | yes | no |
| dew.tools | yes | RDAP 200 | NameCheap, Inc. | 2026-06-17 | no DNS | none | 9.78 / 29.35 | yes | no |
| dew.build | no | RDAP 404 | none | none | none | premium at registrars | 26.26 / 26.26 | no | no |
| dew.training | no | RDAP 404 | none | none | none | premium at registrars | 11.84 / 33.47 | yes | no |
| dew.systems | yes | RDAP or registry WHOIS 200 (17:00-17:40 UTC pass, unchanged) | NameCheap, Inc. | 2027-02-15 | Namecheap parking page | registrar parking | 11.84 / 28.32 | yes | no |
| dew.codes | yes | RDAP or registry WHOIS 200 (17:00-17:40 UTC pass, unchanged) | Name.com, Inc. | 2027-05-29 | GitHub Pages, "Kevin Yong, Portfolio" | in use | 4.63 / 57.16 | yes | no |
| dew.science | no | RDAP 404 | none | none | none | none | 10.79 / 10.79 | yes | no |
| dew.engineering | no | RDAP 404 | none | none | none | none | 6.69 / 52.01 | yes | no |
| dew.study | no | RDAP 404 | none | none | none | none | 1.54 / 31.41 | no | no |
| dew.zone | yes | RDAP 200 | NameCheap, Inc. | 2030-05-16 | HTTP 200, no title | none | 8.24 / 31.41 | yes | no |
| dew.cloud | yes | RDAP 200 | GoDaddy | 2027-02-16 | HTTP 200, no title | none | 3.88 / 21.11 | yes | no |
| dew.network | no | RDAP 404 | none | none | none | none | 4.63 / 28.32 | yes | no |
| dew.software | yes | RDAP 200 | NameCheap, Inc. | 2026-06-17 | no DNS | none | 15.96 / 33.47 | yes | no |
| dew.technology | yes | RDAP 200 | GoDaddy.com, LLC | 2026-11-10 | HTTP 200, no title | none | 9.78 / 23.17 | yes | no |
| dew.tech | no | RDAP 404; registry WHOIS confirms available | none | none | none | none | 6.99 / 50.98 | yes | no |
| dew.foundation | no | RDAP 404 | none | none | none | none | 5.99 / 22.66 | yes | no |
| dew.institute | no | RDAP 404 | none | none | none | none | 7.72 / 22.14 | yes | no |
| dew.academy | yes | RDAP 200 | united-domains GmbH | 2026-11-01 | "dew involvement GmbH — coming soon" | none | 11.84 / 37.59 | yes | no |
| dew.works | no | RDAP 404 | none | none | none | none | 4.63 / 31.41 | yes | no |
| dew.team | no | RDAP 404 | none | none | none | none | 4.63 / 29.35 | yes | no |
| dew.place | no | RDAP 404 | none | none | none | none | 18.02 / 18.02 | yes | no |
| dew.center | no | RDAP 404 | none | none | none | none | 3.60 / 26.26 | yes | no |
| dew.digital | no | RDAP 404 | none | none | none | none | 2.57 / 33.47 | yes | no |
| dew.plus | yes | RDAP 200 | Spaceship, Inc. | 2026-11-04 | "dew.plus for sale \| Spaceship.com" | buy now 1,977 (Spaceship) | 9.78 / 43.77 | yes | no |
| dew.pro | no | RDAP 404 | none | none | none | none | 3.09 / 22.14 | yes | no |
| dew.one | yes | RDAP 200 | Dynadot Inc | 2026-11-28 | HTTP 200, no title | buy now 5,950 (Afternic) | 6.69 / 20.08 | no | no |
| dew.how | no | RDAP 404 | none | none | none | none | 21.11 / 21.11 | yes | no |
| dew.ink | yes | RDAP 200 | Alibaba Cloud Computing Ltd. d/b/a HiChina (www.net.cn) | 2030-09-22 | no DNS | none | 2.06 / 26.26 | yes | no |
| dew.art | no | RDAP 404 | none | none | none | none | 3.60 / 21.11 | no | no |
| dew.design | yes | RDAP 200 | Sav.com LLC | 2027-08-04 | HTTP 200, no title | buy now 2,500 (Afternic) | 10.81 / 46.86 | yes | no |
| dew.studio | yes | RDAP 200 | GoDaddy.com, LLC | 2026-12-02 | "Home \| Dew" | none | 11.84 / 32.44 | yes | no |
| dew.space | no | RDAP 404; registry WHOIS confirms available | none | none | none | none | 1.96 / 26.26 | yes | no |
| dew.world | yes | RDAP 200 | Squarespace Domains II LLC | 2027-05-10 | "Dew.world" | none | 2.57 / 33.47 | yes | no |
| dew.live | yes | RDAP 200 | Name.com, Inc. | 2027-07-31 | no DNS | none | 2.57 / 26.26 | yes | no |
| dew.site | no | RDAP 404; registry WHOIS confirms available | none | none | none | reserved by registry | 1.96 / 28.84 | yes | no |
| dew.online | no | RDAP 404; registry WHOIS confirms available | none | none | none | none | 1.96 / 28.84 | yes | no |
| dew.fyi | yes | RDAP 200 | Porkbun LLC | 2027-08-31 | "dew.fyi — Coming Soon" | none | 5.66 / 5.66 | yes | no |
| dew.link | no | RDAP 404; registry WHOIS confirms available | none | none | none | none | 7.72 / 7.72 | no | no |
| dew.click | no | RDAP 404; registry WHOIS confirms available | none | none | none | none | 1.54 / 10.81 | no | no |
| dew.wiki | yes | RDAP 200 | GoDaddy.com, LLC | 2026-12-19 | no DNS | none | 2.06 / 26.26 | yes | no |
| dew.guru | no | RDAP 404 | none | none | none | none | 2.57 / 34.50 | yes | no |
| dew.expert | yes | RDAP 200 | GoDaddy.com, LLC | 2026-09-17 | HTTP 200, no title | buy now 750 (Afternic) | 6.69 / 49.95 | yes | no |
| dew.so | yes | registry WHOIS 200 | NameCheap | 2027-07-14 | "dew.so for sale \| Spaceship.com" | buy now 5,990 (Spaceship) | not carried | no | no |
| dew.is | yes | RDAP 200 | ? | ? | HTTP 200, no title | none | not carried | no | no |
| dew.in | yes | RDAP 200 | Dynadot, LLC | 2031-02-18 | no DNS | none | 7.83 / 7.83 | no | no |
| dew.us | yes | registry WHOIS 200 | CSC Corporate Domains, Inc. | 2028-04-18 | DNS only, no HTTP | none | 4.43 / 7.00 | yes | no |
| dew.me | no | registry WHOIS: premium, contact premium@identity.digital | none | none | none | registry premium | 2.73 / 17.27 | yes | no |
| dew.cc | yes | RDAP 200 | Porkbun LLC | 2027-02-07 | "dew.cc" | none | 3.40 / 8.55 | yes | no |
| dew.tv | no | RDAP 404 but registry WHOIS says "Reserved Domain Name" | none | none | none | reserved by registry | 26.26 / 26.26 | yes | no |
| dew.to | yes | RDAP 200 | Government of Kingdom of Tonga | 2080-03-20 | no DNS | none | 51.80 / 51.80 | no | no |
| dew.la | yes | registry WHOIS 200 | GoDaddy.com, Inc. | 2027-07-31 | "D.E.W. L.A. , Operations, Marketing and Web Development" | none | 27.34 / 27.94 | no | no |
| dew.gg | yes | registry WHOIS 200 | NameCheap, Inc (https://www.namecheap.com) | ? | "Dew - #1 NFT Aggregator on Polygon" | none | 51.80 / 51.80 | no | no |
| dew.fm | yes | RDAP 200 | Name.com, Inc. | 2027-01-10 | HTTP 404, no title | none | 87.85 / 87.85 | yes | no |
| dew.am | no | registry WHOIS: not found | none | none | none | none | 36.35 / 36.35 | no | no |
| dew.ax | no | registry WHOIS: not found | none | none | none | none | not carried | no | no |
| dew.ee | yes | registry WHOIS 200 | name:       NETIM | 2026-11-05 | HTTP 200, no title | none | not carried | no | no |
| dew.ws | no | registry WHOIS: not found | none | none | none | none | 22.97 / 22.97 | no | no |
| dew.eu | yes | registry WHOIS 200 | Name: Frankcom EU Service | ? | "dew.eu is for sale" | for sale | 5.88 / 5.88 | no | no |

## Appendix B: every other stem

| Domain | Registered | Evidence | Registrar | Expires | Live site or lander | Sale price | Porkbun first year / renewal | Cloudflare | HTTPS forced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dew-framework.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| dew-framework.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dew-framework.com | no | RDAP 404 | none | none | none | none | 11.08 / 11.08 | yes | no |
| dew-framework.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dew-framework.in | no | RDAP 404 | none | none | none | none | 7.83 / 7.83 | no | no |
| dew-framework.io | no | registry WHOIS: not found | none | none | none | none | 28.12 / 51.80 | yes | no |
| dew-framework.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dew-framework.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dew-framework.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dew-framework.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dew-framework.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| dew-ml.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| dew-ml.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dew-ml.com | no | RDAP 404 | none | none | none | none | 11.08 / 11.08 | yes | no |
| dew-ml.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dew-ml.in | no | RDAP 404; registry WHOIS confirms available | none | none | none | none | 7.83 / 7.83 | no | no |
| dew-ml.io | no | registry WHOIS: not found | none | none | none | none | 28.12 / 51.80 | yes | no |
| dew-ml.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dew-ml.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dew-ml.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dew-ml.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dew-ml.sh | no | registry WHOIS: not found | none | none | none | none | 31.20 / 46.65 | no | no |
| dew-ml.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| dewbase.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewbase.com | yes | RDAP 200 | NameCheap, Inc. | 2027-01-15 | no DNS | none | 11.08 / 11.08 | yes | no |
| dewbase.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewbase.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewbase.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewcore.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewcore.com | yes | RDAP 200 | Amazon Registrar, Inc. | 2026-11-10 | no DNS | none | 11.08 / 11.08 | yes | no |
| dewcore.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewcore.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewcore.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewdev.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| dewdev.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewdev.com | yes | RDAP 200 | eName Technology Co., Ltd. | 2027-03-08 | "502 Bad Gateway" | none | 11.08 / 11.08 | yes | no |
| dewdev.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewdev.in | no | RDAP 404; registry WHOIS confirms available | none | none | none | none | 7.83 / 7.83 | no | no |
| dewdev.io | no | registry WHOIS: not found | none | none | none | none | 28.12 / 51.80 | yes | no |
| dewdev.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewdev.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewdev.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewdev.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewdev.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| dewdrop.ai | yes | RDAP 200 | NameCheap, Inc. | 2027-08-25 | HTTP 200, no title | parked on afternic | 82.70 / 82.70 | yes | no |
| dewdrop.app | yes | RDAP 200 | Sav.com, LLC - 34 | 2026-09-14 | "Dewdrop.app for sale \| Spaceship.com" | buy now 999 (Spaceship) | 8.75 / 14.93 | yes | yes |
| dewdrop.com | yes | RDAP 200 | GoDaddy.com, LLC | 2027-03-06 | HTTP 200, no title | parked on afternic | 11.08 / 11.08 | yes | no |
| dewdrop.dev | yes | RDAP 200 | Squarespace Domains II LLC. | 2027-08-11 | HTTP 503, no title | none | 8.75 / 12.87 | yes | yes |
| dewdrop.fyi | no | RDAP 404 | none | none | none | none | 5.66 / 5.66 | yes | no |
| dewdrop.in | yes | RDAP 200 | Spaceship, Inc. | 2027-01-22 | "Just a moment..." | parked on sedo | 7.83 / 7.83 | no | no |
| dewdrop.ink | no | RDAP 404 | none | none | none | none | 2.06 / 26.26 | yes | no |
| dewdrop.io | yes | registry WHOIS 200 | NameCheap, Inc. | 2026-11-09 | HTTP 200, no title | none | 28.12 / 51.80 | yes | no |
| dewdrop.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewdrop.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewdrop.org | yes | RDAP 200 | Spaceship, Inc. | 2027-07-31 | HTTP 200, no title | parked on afternic | 7.98 / 11.84 | yes | no |
| dewdrop.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewdrop.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| dewflow.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewflow.com | yes | RDAP 200 | NameCheap, Inc. | 2027-01-20 | no DNS | none | 11.08 / 11.08 | yes | no |
| dewflow.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewflow.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewflow.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewframework.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| dewframework.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewframework.com | no | RDAP 404 | none | none | none | none | 11.08 / 11.08 | yes | no |
| dewframework.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewframework.in | no | RDAP 404 | none | none | none | none | 7.83 / 7.83 | no | no |
| dewframework.io | no | registry WHOIS: not found | none | none | none | none | 28.12 / 51.80 | yes | no |
| dewframework.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewframework.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewframework.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewframework.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewframework.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| dewgrad.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewgrad.com | no | RDAP 404 | none | none | none | none | 11.08 / 11.08 | yes | no |
| dewgrad.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewgrad.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewgrad.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewhq.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| dewhq.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewhq.com | yes | RDAP 200 | Register SPA | 2028-08-30 | HTTP 200, no title | buy now 2,888 (Afternic) | 11.08 / 11.08 | yes | no |
| dewhq.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewhq.in | no | RDAP 404; registry WHOIS confirms available | none | none | none | none | 7.83 / 7.83 | no | no |
| dewhq.io | no | registry WHOIS: not found | none | none | none | none | 28.12 / 51.80 | yes | no |
| dewhq.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewhq.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewhq.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewhq.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewhq.sh | no | registry WHOIS: not found | none | none | none | none | 31.20 / 46.65 | no | no |
| dewhq.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| dewjax.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| dewjax.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewjax.com | no | RDAP 404 | none | none | none | none | 11.08 / 11.08 | yes | no |
| dewjax.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewjax.in | no | RDAP 404; registry WHOIS confirms available | none | none | none | none | 7.83 / 7.83 | no | no |
| dewjax.io | no | registry WHOIS: not found | none | none | none | none | 28.12 / 51.80 | yes | no |
| dewjax.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewjax.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewjax.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewjax.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewjax.sh | no | registry WHOIS: not found | none | none | none | none | 31.20 / 46.65 | no | no |
| dewjax.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| dewkit.app | yes | RDAP 200 | Namecheap Inc. | 2026-12-20 | no DNS | none | 8.75 / 14.93 | yes | yes |
| dewkit.com | yes | RDAP 200 | DNSPod, Inc. | 2026-11-18 | no DNS | none | 11.08 / 11.08 | yes | no |
| dewkit.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewkit.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewkit.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewlab.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| dewlab.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewlab.com | yes | RDAP 200 | TurnCommerce, Inc. DBA NameBright.com | 2026-09-25 | "DewLab.com is for sale \| HugeDomains" | buy now 395.63 (HugeDomains) | 11.08 / 11.08 | yes | no |
| dewlab.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewlab.in | no | RDAP 404; registry WHOIS confirms available | none | none | none | none | 7.83 / 7.83 | no | no |
| dewlab.io | yes | registry WHOIS 200 | Cloudflare, Inc | 2027-02-01 | no DNS | none | 28.12 / 51.80 | yes | no |
| dewlab.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewlab.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewlab.org | yes | RDAP 200 | Tucows Domains Inc. | 2029-05-05 | "DEW Lab \| Teachers College" | none | 7.98 / 11.84 | yes | no |
| dewlab.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewlab.sh | no | registry WHOIS: not found | none | none | none | none | 31.20 / 46.65 | no | no |
| dewlab.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| dewlabs.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| dewlabs.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewlabs.com | yes | RDAP 200 | TurnCommerce, Inc. DBA NameBright.com | 2026-11-29 | "DewLabs.com is for sale \| HugeDomains" | buy now 100 (HugeDomains) | 11.08 / 11.08 | yes | no |
| dewlabs.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewlabs.in | no | RDAP 404 | none | none | none | none | 7.83 / 7.83 | no | no |
| dewlabs.io | yes | registry WHOIS 200 | Cloudflare, Inc | 2027-02-12 | "Dew Labs – Dew point based humidity control" | none | 28.12 / 51.80 | yes | no |
| dewlabs.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewlabs.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewlabs.org | yes | RDAP 200 | Tucows Domains Inc. | 2027-03-30 | "DEW - DEW" | none | 7.98 / 11.84 | yes | no |
| dewlabs.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewlabs.sh | no | registry WHOIS: not found | none | none | none | none | 31.20 / 46.65 | no | no |
| dewlabs.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| dewline.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewline.com | yes | RDAP 200 | GoDaddy.com, LLC | 2027-05-15 | no DNS | none | 11.08 / 11.08 | yes | no |
| dewline.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewline.org | yes | RDAP 200 | Domain.com - Network Solutions, LLC | 2026-12-11 | no DNS | none | 7.98 / 11.84 | yes | no |
| dewline.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewloop.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewloop.com | yes | RDAP 200 | Alibaba Cloud Computing Ltd. d/b/a HiChina (www.net.cn) | 2027-07-22 | no DNS | none | 11.08 / 11.08 | yes | no |
| dewloop.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewloop.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewloop.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewml.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| dewml.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewml.cloud | no | RDAP 404; registry WHOIS confirms available | none | none | none | none | 3.88 / 21.11 | yes | no |
| dewml.com | no | RDAP 404 | none | none | none | none | 11.08 / 11.08 | yes | no |
| dewml.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewml.in | no | RDAP 404; registry WHOIS confirms available | none | none | none | none | 7.83 / 7.83 | no | no |
| dewml.io | no | registry WHOIS: not found | none | none | none | none | 28.12 / 51.80 | yes | no |
| dewml.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewml.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewml.net | no | RDAP 404 | none | none | none | none | 12.52 / 12.52 | yes | no |
| dewml.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewml.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewml.run | no | RDAP 404 | none | none | none | none | 4.12 / 22.14 | yes | no |
| dewml.sh | no | registry WHOIS: not found | none | none | none | none | 31.20 / 46.65 | no | no |
| dewml.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| dewml.tech | no | RDAP 404; registry WHOIS confirms available | none | none | none | none | 6.99 / 50.98 | yes | no |
| dewml.tools | no | RDAP 404 | none | none | none | none | 9.78 / 29.35 | yes | no |
| dewpoint.ai | yes | RDAP 200 | Virtualia LLC | 2028-01-02 | no DNS | none | 82.70 / 82.70 | yes | no |
| dewpoint.app | yes | RDAP 200 | Squarespace Domains II LLC. | 2027-05-09 | "Domains for sale" | for sale | 8.75 / 14.93 | yes | yes |
| dewpoint.com | yes | RDAP 200 | Network Solutions, LLC | 2029-01-29 | "Managed IT, Cybersecurity and Technology Consulting \| Dewpoint" | none | 11.08 / 11.08 | yes | no |
| dewpoint.dev | yes | RDAP 200 | CloudFlare, Inc. | 2027-05-30 | no DNS | none | 8.75 / 12.87 | yes | yes |
| dewpoint.in | yes | RDAP 200 | Rediff.com India Limited | 2027-11-26 | "Indoor Air Quality Consultants and Smart HVAC Solutions" | none | 7.83 / 7.83 | no | no |
| dewpoint.io | yes | registry WHOIS 200 | Dynadot Inc | 2026-12-14 | HTTP 200, no title | none | 28.12 / 51.80 | yes | no |
| dewpoint.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewpoint.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewpoint.org | yes | RDAP 200 | GoDaddy.com, LLC | 2029-09-12 | HTTP 200, no title | parked on afternic | 7.98 / 11.84 | yes | no |
| dewpoint.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewpoint.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| dewrun.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewrun.com | yes | RDAP 200 | GoDaddy.com, LLC | 2027-01-06 | no DNS | parked on afternic | 11.08 / 11.08 | yes | no |
| dewrun.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewrun.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewrun.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewstack.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewstack.com | yes | RDAP 200 | NameCheap, Inc. | 2026-11-22 | no DNS | none | 11.08 / 11.08 | yes | no |
| dewstack.dev | yes | RDAP 200 | Namecheap Inc. | 2027-09-01 | no DNS | none | 8.75 / 12.87 | yes | yes |
| dewstack.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewstack.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewtrain.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| dewtrain.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewtrain.com | no | RDAP 404 | none | none | none | none | 11.08 / 11.08 | yes | no |
| dewtrain.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewtrain.in | no | RDAP 404 | none | none | none | none | 7.83 / 7.83 | no | no |
| dewtrain.io | no | registry WHOIS: not found | none | none | none | none | 28.12 / 51.80 | yes | no |
| dewtrain.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewtrain.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewtrain.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewtrain.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewtrain.sh | no | registry WHOIS: not found | none | none | none | none | 31.20 / 46.65 | no | no |
| dewtrain.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| dewtraining.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| dewtraining.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| dewtraining.com | no | RDAP 404 | none | none | none | none | 11.08 / 11.08 | yes | no |
| dewtraining.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| dewtraining.in | no | RDAP 404 | none | none | none | none | 7.83 / 7.83 | no | no |
| dewtraining.io | no | registry WHOIS: not found | none | none | none | none | 28.12 / 51.80 | yes | no |
| dewtraining.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewtraining.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewtraining.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| dewtraining.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewtraining.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| dewy.ai | yes | RDAP 200 | NameCheap, Inc. | 2027-06-21 | "Dewy.ai for sale \| Spaceship.com" | buy now 167,881 (Spaceship) | 82.70 / 82.70 | yes | no |
| dewy.app | yes | RDAP 200 | Sav.com, LLC | 2026-10-29 | HTTP 200, no title | parked on afternic | 8.75 / 14.93 | yes | yes |
| dewy.com | yes | RDAP 200 | NameSilo, LLC | 2027-02-11 | DNS only, no HTTP | none | 11.08 / 11.08 | yes | no |
| dewy.dev | yes | RDAP 200 | CloudFlare, Inc. | 2027-01-13 | no DNS | none | 8.75 / 12.87 | yes | yes |
| dewy.fyi | no | RDAP 404 | none | none | none | none | 5.66 / 5.66 | yes | no |
| dewy.in | yes | RDAP 200 | GoDaddy | 2028-01-16 | HTTP 200, no title | none | 7.83 / 7.83 | no | no |
| dewy.ink | no | RDAP 404 | none | none | none | none | 2.06 / 26.26 | yes | no |
| dewy.io | yes | registry WHOIS 200 | GoDaddy.com, LLC | 2027-07-12 | "10x your patient follow-up with AI \| Dewy" | live product, 319 shown on page | 28.12 / 51.80 | yes | no |
| dewy.is | yes | RDAP 200 | ? | ? | "Store unavailable" | none | not carried by Porkbun | no | no |
| dewy.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| dewy.org | yes | RDAP 200 | GoDaddy.com, LLC | 2027-05-26 | "DEWY.ORG - For Sale" | for sale | 7.98 / 11.84 | yes | no |
| dewy.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| dewy.sh | no | registry WHOIS: not found | none | none | none | none | 31.20 / 46.65 | no | no |
| dewy.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| getdew.ai | yes | RDAP 200 | GoDaddy.com, LLC | 2028-07-14 | HTTP 200, no title | none | 82.70 / 82.70 | yes | no |
| getdew.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| getdew.com | yes | RDAP 200 | GoDaddy.com, LLC | 2026-10-23 | "Just a moment..." | parked on atom.com | 11.08 / 11.08 | yes | no |
| getdew.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| getdew.in | no | RDAP 404 | none | none | none | none | 7.83 / 7.83 | no | no |
| getdew.io | yes | registry WHOIS 200 | NameCheap, Inc. | 2027-06-19 | HTTP 200, no title | none | 28.12 / 51.80 | yes | no |
| getdew.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| getdew.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| getdew.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| getdew.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| getdew.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| jaxdew.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| jaxdew.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| jaxdew.com | no | RDAP 404 | none | none | none | none | 11.08 / 11.08 | yes | no |
| jaxdew.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| jaxdew.in | no | RDAP 404 | none | none | none | none | 7.83 / 7.83 | no | no |
| jaxdew.io | no | registry WHOIS: not found | none | none | none | none | 28.12 / 51.80 | yes | no |
| jaxdew.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| jaxdew.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| jaxdew.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| jaxdew.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| jaxdew.sh | no | registry WHOIS: not found | none | none | none | none | 31.20 / 46.65 | no | no |
| jaxdew.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| mldew.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| mldew.com | yes | RDAP 200 | Gname.com Pte. Ltd. | 2026-10-30 | no DNS | none | 11.08 / 11.08 | yes | no |
| mldew.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| mldew.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| mldew.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| morningdew.ai | yes | RDAP 200 | NameCheap, Inc. | 2027-02-13 | HTTP 200, no title | none | 82.70 / 82.70 | yes | no |
| morningdew.app | no | RDAP 404 | none | none | none | none | 8.75 / 14.93 | yes | yes |
| morningdew.com | yes | RDAP 200 | Dynadot Inc | 2027-10-11 | "MorningDew.com for sale" | for sale | 11.08 / 11.08 | yes | no |
| morningdew.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| morningdew.in | yes | RDAP 200 | NIXI Holding Account | 2026-12-23 | "morningdew.in - Domain For Sale" | for sale, 499 shown | 7.83 / 7.83 | no | no |
| morningdew.ink | yes | RDAP 200 | NameSilo, LLC | 2027-06-10 | "Site is created successfully!" | none | 2.06 / 26.26 | yes | no |
| morningdew.io | yes | registry WHOIS 200 | GoDaddy.com, LLC | 2026-12-08 | HTTP 200, no title | none | 28.12 / 51.80 | yes | no |
| morningdew.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| morningdew.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| morningdew.org | yes | RDAP 200 | NameCheap, Inc. | 2027-11-07 | DNS only, no HTTP | none | 7.98 / 11.84 | yes | no |
| morningdew.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| morningdew.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| trydew.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| trydew.app | yes | RDAP 200 | CloudFlare, Inc. | 2027-05-26 | "Dew — Private period tracker for iPhone" | for sale | 8.75 / 14.93 | yes | yes |
| trydew.com | yes | RDAP 200 | GoDaddy.com, LLC | 2026-09-17 | HTTP 200, no title | parked on afternic | 11.08 / 11.08 | yes | no |
| trydew.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| trydew.in | no | RDAP 404 | none | none | none | none | 7.83 / 7.83 | no | no |
| trydew.io | no | registry WHOIS: not found | none | none | none | none | 28.12 / 51.80 | yes | no |
| trydew.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| trydew.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| trydew.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| trydew.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| trydew.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
| usedew.ai | no | RDAP 404 | none | none | none | none | 82.70 / 82.70 | yes | no |
| usedew.app | yes | RDAP 200 | Squarespace Domains II LLC. | 2027-01-13 | "Dew — AI Image Generation Studio" | none | 8.75 / 14.93 | yes | yes |
| usedew.com | yes | RDAP 200 | Spaceship, Inc. | 2027-02-16 | HTTP 200, no title | parked on afternic | 11.08 / 11.08 | yes | no |
| usedew.dev | no | RDAP 404 | none | none | none | none | 8.75 / 12.87 | yes | yes |
| usedew.in | no | RDAP 404 | none | none | none | none | 7.83 / 7.83 | no | no |
| usedew.io | no | registry WHOIS: not found | none | none | none | none | 28.12 / 51.80 | yes | no |
| usedew.is | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| usedew.ml | no | RDAP 404 | none | none | none | none | not carried by Porkbun | no | no |
| usedew.org | no | RDAP 404 | none | none | none | none | 7.98 / 11.84 | yes | no |
| usedew.page | no | RDAP 404 | none | none | none | none | 10.81 / 10.81 | yes | yes |
| usedew.so | no | registry WHOIS: not found | none | none | none | none | not carried by Porkbun | no | no |
