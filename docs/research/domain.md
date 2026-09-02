# Domain recommendation for Dew

Measured 2026-09-02 (17:00 to 17:40 UTC). Every registration fact below comes from the authoritative RDAP or WHOIS server for that TLD, and every price from a registrar page or public price API fetched in this session. Aggregator sites were not used: one of them had already reported `dew.sh` as available, and the registry says it is registered (see the availability table).

## Recommendation

Register `dewml.dev` today. Every exact-match `dew.<tld>` that developers would guess is taken: `dew.dev` runs an unrelated live site, `dew.io`, `dew.org`, `dew.ai` and `dew.xyz` are parked for sale at 75,000 to 199,000 USD, and `dew.app`, `dew.sh`, `dew.co`, `dew.run`, `dew.net`, `dew.systems` and `dew.codes` are held by others. The exact matches that are still open (`dew.page`, `dew.day`, `dew.training`, `dew.build`, `dew.ml`) are all registry premium names, so their higher price recurs every year and the registry can change it. `dewml.dev` is the PyPI distribution name `dew-ml` without the hyphen, so the domain, the package and the `import dew` name line up. It is a standard-price `.dev` (10 to 16 USD a year depending on registrar; Cloudflare charges the registry price), `.dev` is on Cloudflare Registrar, and the whole TLD is HSTS-preloaded in Chromium, so the docs site is HTTPS-only with no configuration. The best exact-match alternative is `dew.page`: it is also HSTS-preloaded and on Cloudflare, but it is a premium name at 65 to 77 USD a year for as long as it is held. `dew.training` says what the framework does but costs 73 to 86 USD a year and is 12 characters. `dew.tools` is in `pendingDelete` at the registry and may drop within days; it is worth a daily check but not a plan.

## Ranked table

Scores per criterion: 2 good, 1 mixed, 0 poor. Brand match uses 3 for the bare word `dew`, 2 for `dew` plus the package descriptor (`ml`), 1 for `dew` plus a generic word, 0 for a verb prefix. Only names that can be registered today (or `dew.tools` once it drops) are ranked, in order of total. Aftermarket names are listed in the availability table and skipped here. The last column is the judgment; the totals alone do not separate a cheap `.dev` that adds nothing from one that carries the package name.

| Rank | Domain | Brand match | Readability | Dev-tool signal | HTTPS/HSTS | Price and renewal | On Cloudflare Registrar | Confusion risk | Total | Buy now or defer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | dewml.dev | 2 | 1 | 2 | 2 (preloaded) | 2 (standard, 12.87/yr renewal at Porkbun) | 2 (.dev listed) | 2 | 13 | Buy now |
| 2 | dew-ml.dev | 2 | 1 (hyphen) | 2 | 2 (preloaded) | 2 | 2 | 2 | 13 | Optional, as a redirect for the typed package name |
| 3 | dew.tools (only if it drops) | 3 | 2 | 1 | 1 | 2 (standard .tools, 29.35/yr renewal at Porkbun) | 2 (.tools listed) | 1 | 12 | Watch; not registrable today |
| 4 | getdew.dev | 0 | 2 | 2 | 2 (preloaded) | 2 | 2 | 2 | 12 | Skip; verb prefix reads as a SaaS landing page |
| 5 | trydew.dev, usedew.dev | 0 | 2 | 2 | 2 (preloaded) | 2 | 2 | 2 | 12 | Skip; same reason |
| 6 | dewlabs.dev | 1 | 2 | 2 | 2 (preloaded) | 2 | 2 | 1 (implies an organisation that does not exist) | 12 | Skip |
| 7 | dewframework.dev | 1 | 1 (16 characters) | 2 | 2 (preloaded) | 2 | 2 | 2 | 12 | Skip |
| 8 | dew.page | 3 | 2 | 1 | 2 (preloaded) | 1 (premium, 64.93 to 76.70/yr, recurring) | 1 (.page listed; premium tier must be confirmed in the dashboard) | 1 (dew.dev is someone else's live site) | 11 | Defer; best exact match if one is wanted |
| 9 | dew.day | 3 | 2 | 0 (.day is marketed for dates and events) | 2 (preloaded) | 1 (premium, 64.93 to 76.70/yr) | 1 | 1 | 10 | Skip |
| 10 | dew.training | 3 | 1 (12 characters) | 1 (says training, but the TLD is marketed for courses) | 1 (not preloaded; set HSTS yourself) | 1 (premium, 72.57 to 85.80/yr) | 1 (.training listed; premium tier must be confirmed) | 1 | 9 | Defer; second exact match |
| 11 | dew.build | 3 | 2 | 1 | 1 | 0 (premium, 164.27 to 195.00/yr) | 0 (.build not listed) | 1 | 8 | Skip |
| 12 | dew.ml | 3 | 1 | 0 (Mali ccTLD; `ml` reads as machine learning only to insiders) | 1 | 0 (registry premium, price not offered by Cloudflare or Porkbun) | 0 (.ml not listed) | 0 | 5 | Skip |

Tie-breaks and reading the totals. `dewml.dev` over `dew-ml.dev`: people drop hyphens when they type or say a name, and the package imports as `dew`, not `dew_ml`. `dewml.dev` over `getdew.dev` and the other verb-prefix names: they score 12 on cost and safety alone, but none carries the package name, and a second cheap `.dev` adds a renewal without adding identity. `dew.page` is the highest-scoring exact match; it loses to the `.dev` names only on price, premium handling and the `dew.dev` collision. `dew.page` over `dew.day`: `.page` is neutral, `.day` is a mismatch for a training framework. `dew.tools` is scored but cannot be bought today; it is in `pendingDelete` (see below).

## Availability table

RDAP servers come from the IANA bootstrap file `https://data.iana.org/rdap/dns.json` (publication 2026-07-23T02:00:03Z). It maps `dev`, `app`, `page`, `day` to `https://pubapi.registry.google/rdap/`; `ai`, `run`, `tools`, `training`, `systems`, `codes` to `https://rdap.identitydigital.services/rdap/`; `build` to `https://rdap.centralnic.com/build/`; `xyz` to `https://rdap.centralnic.com/xyz/`; `ml` to `https://rdap.nic.ml/`; `org` to `https://rdap.publicinterestregistry.org/rdap/`; `net` to `https://rdap.verisign.com/net/v1/`. The file has no entry for `io`, `sh` or `co`. For those three the IANA root zone database (`https://www.iana.org/domains/root/db/io.html`, `sh.html`, `co.html`) names the registry WHOIS servers `whois.nic.io`, `whois.nic.sh` and `whois.registry.co`, and those port-43 servers were queried directly. `whois.nic.io` and `whois.nic.sh` state that "the data in this record is provided by Identity Digital or the Registry Operator", and the Identity Digital RDAP server returns the same records; `whois.registry.co` states that "the Whois and RDAP services are provided by CentralNic", and `https://rdap.registry.co/co/domain/dew.co` returns the same record. A 404 from an RDAP server means no registration object exists; the registrar rows in the price table show whether such a name is standard or premium.

Live-site checks are DNS A lookup, HTTP HEAD on `https://` and `http://`, and the HTML `<title>` of a GET. "No DNS" means no A record.

| Domain | Registered | Evidence (RDAP or WHOIS) | Registrar | Registered / expires | Live site | Parked or for sale |
| --- | --- | --- | --- | --- | --- | --- |
| dew.dev | Yes | `pubapi.registry.google/rdap/domain/dew.dev` 200, status `client transfer prohibited` | Dynadot LLC | 2024-07-17 / 2027-07-17 | HTTPS 200, title "DewDev", nameservers `alla.ns.cloudflare.com`, `dan.ns.cloudflare.com` | In use by an unrelated site; not for sale |
| dew.ai | Yes | `rdap.identitydigital.services/rdap/domain/dew.ai` 200, status `active` | NameCheap, Inc. | 2017-12-16 / 2027-04-11 | HTTPS 200, title "dew.ai for sale, Spaceship.com", nameservers `launch1.spaceship.net`, `launch2.spaceship.net` | For sale, offer only; the lander config sets `minOfferPrice` 99,500 USD |
| dew.io | Yes | `whois.nic.io` port 43: Registry Expiry Date 2026-11-18, status `clientDeleteProhibited`, `clientRenewProhibited`, `clientTransferProhibited`, `clientUpdateProhibited`; same record from `rdap.identitydigital.services/rdap/domain/dew.io` | GoDaddy.com, LLC | 2016-11-18 / 2026-11-18 | HTTPS 200; body is a JavaScript redirect to `/lander`, which redirects to `forsale.godaddy.com/forsale/dew.io`; nameservers `ns3.afternic.com`, `ns4.afternic.com` | For sale at GoDaddy: 74,995 USD, or lease 1,625 USD a month |
| dew.sh | Yes | `whois.nic.sh` port 43: Creation Date 2023-09-25, Registry Expiry Date 2027-09-25, status `clientTransferProhibited`; same record from `rdap.identitydigital.services/rdap/domain/dew.sh` | NameCheap, Inc. | 2023-09-25 / 2027-09-25 | HTTPS 200 on Vercel, title "DEW.SH - A Dub Custom Domain" | In use as a Dub short-link domain; not for sale. The aggregator result "available" was wrong |
| dew.app | Yes | `pubapi.registry.google/rdap/domain/dew.app` 200, status `client delete prohibited`, `client transfer prohibited`, `client update prohibited` | Namecamp Limited (Yay.com) | 2018-05-08 / 2027-05-08 | HTTPS 200, title "Domain name registration - dew.app", body "This domain is registered with Yay.com" | Registrar holding page; no sale listing |
| dew.page | No | `pubapi.registry.google/rdap/domain/dew.page` 404 "Not Found" | none | none | No DNS | Registry premium (Porkbun and Namecheap both flag it) |
| dew.day | No | `pubapi.registry.google/rdap/domain/dew.day` 404 "Not Found" | none | none | No DNS | Registry premium (Porkbun and Namecheap both flag it) |
| dew.run | Yes | `rdap.identitydigital.services/rdap/domain/dew.run` 200, status `active` | Alibaba Cloud Computing Ltd. d/b/a HiChina | 2023-04-13 / 2027-04-13 | No DNS; nameservers `ns1-4.363.hk` | Dormant; no sale listing found |
| dew.tools | Yes, dropping | `rdap.identitydigital.services/rdap/domain/dew.tools` 200, status `client transfer prohibited`, `pending delete`; expiration 2026-06-17, last changed 2026-08-28T20:34Z, re-checked 2026-09-02T17:38Z still `pending delete` | NameCheap, Inc. | 2025-06-17 / 2026-06-17 (expired) | No DNS | Will be purged "after several days" in `pendingDelete` per ICANN's EPP status page, then open for registration |
| dew.build | No | `rdap.centralnic.com/build/domain/dew.build` 404 "Object not found" | none | none | No DNS | Registry premium (Porkbun and Namecheap both flag it) |
| dew.ml | No | `rdap.nic.ml/domain/dew.ml` 200 with notice "Premium String - Not Registered, Available as a Premium" and variant relation `RESTRICTED_REGISTRATION` | none | none | No DNS | Registry premium; not sold by Cloudflare or Porkbun |
| dew.training | No | `rdap.identitydigital.services/rdap/domain/dew.training` 404 "Object not found" | none | none | No DNS | Registry premium (Porkbun and Namecheap both flag it) |
| dew.systems | Yes | `rdap.identitydigital.services/rdap/domain/dew.systems` 200, status `client transfer prohibited` | NameCheap, Inc. | 2026-02-15 / 2027-02-15 | HTTP redirects to `https://www.dew.systems/`, Namecheap page "has been recently registered with namecheap.com" | Registrar parking page |
| dew.codes | Yes | `rdap.identitydigital.services/rdap/domain/dew.codes` 200, status `client transfer prohibited` | Name.com, Inc. | 2026-05-29 / 2027-05-29 | HTTPS 200 on GitHub Pages, title "Kevin Yong, Portfolio" | In use as a personal site |
| dew.org | Yes | `rdap.publicinterestregistry.org/rdap/domain/dew.org` 200, status `client delete prohibited`, `client renew prohibited`, `client transfer prohibited`, `client update prohibited` | GoDaddy.com, LLC | 2000-03-28 / 2027-03-28 | HTTPS 200; JavaScript redirect to `/lander` then to `forsale.godaddy.com/forsale/dew.org`; nameservers `ns1.afternic.com`, `ns2.afternic.com` | For sale at GoDaddy: 135,000 USD, or lease 4,744 USD a month |
| dew.net | Yes | `rdap.verisign.com/net/v1/domain/dew.net` 200, status `client transfer prohibited` | Net-Chinese Co., Ltd. | 1995-11-25 / 2026-11-25 | No DNS; nameservers `NS1.OFFIS.COM`, `NS2.OFFIS.COM`, `DNS2.OFFIS.COM.AU` | Dormant; no sale listing found |
| dew.co | Yes | `whois.registry.co` port 43: Creation Date 2010-07-20, Registry Expiry Date 2027-07-19, status includes `autoRenewPeriod`; same record from `rdap.registry.co/co/domain/dew.co` | GoDaddy.com, LLC | 2010-07-20 / 2027-07-19 | HTTPS 200 behind Cloudflare, title "Dew of the Gods, Mindful, vegan skincare" | In use by a skincare brand |
| dew.xyz | Yes | `rdap.centralnic.com/xyz/domain/dew.xyz` 200, status `client transfer prohibited` | Dynadot Inc | 2021-09-25 / 2030-09-25 | HTTPS 200, title "Dew.xyz for sale, Spaceship.com" | For sale, buy now 198,795 USD |
| getdew.dev | No | `pubapi.registry.google/rdap/domain/getdew.dev` 404 | none | none | No DNS | Standard price at Porkbun and Namecheap |
| dewml.dev | No | `pubapi.registry.google/rdap/domain/dewml.dev` 404 | none | none | No DNS | Standard price at Porkbun and Namecheap |
| dew-ml.dev | No | `pubapi.registry.google/rdap/domain/dew-ml.dev` 404 | none | none | No DNS | Standard price at Porkbun |
| usedew.dev | No | `pubapi.registry.google/rdap/domain/usedew.dev` 404 | none | none | No DNS | Standard price at Porkbun |
| trydew.dev | No | `pubapi.registry.google/rdap/domain/trydew.dev` 404 | none | none | No DNS | Standard price at Porkbun |
| dewframework.dev | No | `pubapi.registry.google/rdap/domain/dewframework.dev` 404 | none | none | No DNS | Standard price at Porkbun |
| dewlabs.dev | No | `pubapi.registry.google/rdap/domain/dewlabs.dev` 404 (first query was rate-limited with 429; retried after 5 s) | none | none | No DNS | Standard price at Porkbun |

## Price table

Prices in USD, fetched 2026-09-02. Porkbun figures come from the public price API `GET https://api.porkbun.com/api/json/v3/pricing/get` (per-TLD standard prices) and from the search page `https://porkbun.com/checkout/search?q=<domain>` rendered in headless Chrome (per-name premium flags). Namecheap figures come from `https://www.namecheap.com/domains/registration/results/?domain=<domain>` rendered the same way. Cloudflare publishes no price list. Its docs say it charges "the registry and ICANN list price with no markup" (Registrar FAQ, updated 2026-08-03) and that it sells "both standard and premium domains at registry cost" (Cloudflare Learning, "What is a premium domain?"). The Registrar API guide (updated 2026-04-24) shows example quotes of 10.11 for a standard `.dev` and 11.00 for a standard `.app`, and notes that premium registration through the API is not yet supported ("when supported, premium domains will require explicit fee acknowledgement"). The live Cloudflare price appears only in the dashboard search, which sits behind a Turnstile challenge, so it was not captured here.

| Domain | Tier | Porkbun first year / renewal | Namecheap first year / renewal | Cloudflare Registrar |
| --- | --- | --- | --- | --- |
| dewml.dev, dew-ml.dev, getdew.dev, usedew.dev, trydew.dev, dewframework.dev, dewlabs.dev | Standard .dev | 8.75 / 12.87 | 10.98 / 15.98 (getdew.dev, dewml.dev checked) | .dev listed; at cost; doc example 10.11 |
| dew.page | Registry premium | 64.93 / 64.93 | 76.70 / 76.70 | .page listed; premium at registry cost per Cloudflare; confirm the tier in the dashboard |
| dew.day | Registry premium | 64.93 / 64.93 | 76.70 / 76.70 | .day listed; same note |
| dew.training | Registry premium | 36.32 (sale) / 72.57 | 85.80 / 85.80 | .training listed; same note |
| dew.build | Registry premium | 164.27 / 164.27 | 195.00 / 195.00 | .build not on Cloudflare's TLD list |
| dew.ml | Registry premium | .ml not sold by Porkbun | not checked | .ml not on Cloudflare's TLD list |
| dew.tools (after it drops, if standard) | Standard .tools | 9.78 / 29.35 | not checked | .tools listed |
| dew.io | Aftermarket | 74,995 buy now, or 1,625 a month lease (GoDaddy) | | .io listed; renewal 51.80 at Porkbun |
| dew.org | Aftermarket | 135,000 buy now, or 4,744 a month lease (GoDaddy) | | .org listed; renewal 11.84 at Porkbun |
| dew.xyz | Aftermarket | 198,795 buy now (Spaceship) | | .xyz listed; renewal 14.21 at Porkbun |
| dew.ai | Aftermarket | offer only, minimum 99,500 (Spaceship) | | .ai listed; renewal 82.70 at Porkbun |

Standard renewal prices per TLD from the Porkbun API, for reference: dev 12.87, app 14.93, page 10.81, day 10.81, io 51.80, sh 46.65, ai 82.70, run 22.14, tools 29.35, build 26.26, training 33.47, systems 28.32, codes 57.16, org 11.84, net 12.52, co 31.20, xyz 14.21. Google Registry's `.dev` and `.page` pricing policies define a premium name as one that "carries an annual registration fee that is greater than the standard domain registration fee" and let the registry "update the rate card" over time, so a premium price is a recurring cost that can move.

Cloudflare TLD support was read from `https://www.cloudflare.com/tld-policies/`. Listed: ai, app, co, codes, day, dev, io, net, org, page, run, systems, tools, training, xyz. Not listed: build, ml, sh. Cloudflare Registrar domains must use Cloudflare nameservers (Registrar FAQ), which fits the owner's setup.

HTTPS behaviour was checked in Chromium's `net/http/transport_security_state_static.json` (main branch, fetched from `chromium.googlesource.com`). `dev`, `app`, `page` and `day` each have a TLD-level entry with `"mode": "force-https", "include_subdomains": true`. None of the other candidate TLDs has one. Google Registry's `get.dev`, `get.app`, `get.page` and `new.day` pages say the same in prose. On a preloaded TLD the browser rewrites every plain-HTTP request to HTTPS before sending it, so the docs host must serve a valid certificate from day one; on the others HTTPS is optional and HSTS is a header you add yourself.

## What to buy today and what to skip

Buy today:

- `dewml.dev` on Cloudflare Registrar. Standard tier; Porkbun renews it at 12.87 and Namecheap lists 15.98 retail, and Cloudflare charges the registry price shown at checkout. Set it as `site_url` in `mkdocs.yml` and point it at the docs host.
- Optional: `dew-ml.dev`, same price, as a redirect to `dewml.dev`. It catches people who type the PyPI name.

Watch:

- `dew.tools`. It is in `pendingDelete` and the registry will purge it "after several days" (ICANN EPP status page); the last status change was 2026-08-28. Query `https://rdap.identitydigital.services/rdap/domain/dew.tools` once a day; when it returns 404, search it in the Cloudflare dashboard at once. Drop-catch services compete for expiring names, so this may fail. Porkbun's standard `.tools` renewal is 29.35 a year; the registry may also re-price it as premium when it drops, which cannot be known in advance.

Defer, and only if an exact match matters more than cost:

- `dew.page`: 64.93 to 76.70 USD a year, recurring, HSTS-preloaded, on Cloudflare.
- `dew.training`: 72.57 to 85.80 USD a year, recurring, not preloaded, on Cloudflare.

Skip:

- `dew.day` (wrong TLD meaning), `dew.build` (164 to 195 a year and not on Cloudflare), `dew.ml` (Mali ccTLD, premium, not on Cloudflare or Porkbun).
- `getdew.dev`, `trydew.dev`, `usedew.dev`, `dewlabs.dev`, `dewframework.dev`: nothing wrong with them, but none carries the package name and buying several names adds renewals without adding identity.
- All aftermarket names: `dew.io` (74,995), `dew.org` (135,000), `dew.xyz` (198,795), `dew.ai` (offers from 99,500). Not a sane spend for an open-source framework.
- Names in use by others: `dew.dev`, `dew.sh`, `dew.co`, `dew.codes`, `dew.app`, `dew.systems`, `dew.run`, `dew.net`. `dew.net` (expires 2026-11-25, held since 1995) and `dew.io` (expires 2026-11-18, held since 2016) are the only ones that could lapse soon; `dew.io` is actively listed for sale, so do not plan on either.

## Domain versus GitHub and PyPI identity

Checked 2026-09-02:

- GitHub: `https://api.github.com/repos/AshishKumar4/dew` returns the repository ("A nice ML framework based on Jax/Flax"). The login `Dew` is a user account created 2012-09-04, so a `dew` organisation is not available. `dew-ml`, `dewml` and `dew-framework` return 404 from `https://api.github.com/users/<login>`, so those logins are unclaimed.
- PyPI: `pyproject.toml` names the distribution `dew-ml`, but `https://pypi.org/pypi/dew-ml/json` returns 404. The name is not published or reserved. `https://pypi.org/pypi/dew/json` returns a package `dew` 0.0.1 with an empty summary, one release, so the bare name is taken by a placeholder. Publish `dew-ml` (a real early release, not an empty one) before announcing the domain, or the first thing a visitor tries, `pip install dew-ml`, fails.
- npm: `https://registry.npmjs.org/dew` is an active, unrelated project ("Sandboxed Linux compute, agent-native and human-friendly", repository `solcreek/dew`, 42 versions, last modified 2026-07-09). GitHub search also shows `gudaoxuri/dew` (400 stars, a Spring Cloud microservice stack). The bare word `dew` is shared with these and with `dew.dev` (DewDev). None of them touches the JAX or Python space, so the collision is a search-result nuisance, not a naming conflict.

What this means for the domain. The three identities do not need to be the same string. The repo is found through GitHub, the package through PyPI, and the domain only has to be memorable and consistent with them. `dewml.dev` is the PyPI name minus a hyphen, the import stays `dew`, and the README can state all three on one line: `github.com/AshishKumar4/dew`, `pip install dew-ml`, `dewml.dev`. If a GitHub organisation is ever wanted, `dew-ml` is free and matches the package. An exact-match domain such as `dew.page` would not change any of this; it would only add a second string that differs from the package name and a premium renewal.

## Method

- RDAP: `GET <server>/domain/<name>` with `Accept: application/rdap+json`, servers taken from the IANA bootstrap file. Status, registrar (entity with role `registrar`), events (`registration`, `expiration`, `last changed`) and nameservers were read from the JSON.
- WHOIS for io, sh, co: raw port-43 queries to the servers named in the IANA root zone database.
- Live sites: `getaddrinfo`, HTTP HEAD and GET with a 15 s timeout, `<title>` and sale-lander keywords read from the body; GoDaddy landers were rendered in headless Chrome because `forsale.godaddy.com` returns 403 to plain clients.
- Prices: Porkbun public price API and search pages, Namecheap search pages, both rendered in headless Chrome with a 10 to 12 s virtual-time budget; Cloudflare docs for policy statements.
- HSTS: Chromium `transport_security_state_static.json`, searched for TLD-level entries.

Sources: `https://data.iana.org/rdap/dns.json`; `https://www.iana.org/domains/root/db/{io,sh,co}.html`; `https://www.icann.org/resources/pages/epp-status-codes-2014-06-16-en`; `https://www.registry.google/policies/pricing/dev/` and `.../page/`; `https://get.dev/`, `https://get.app/`, `https://get.page/`, `https://new.day/`; `https://chromium.googlesource.com/chromium/src/+/main/net/http/transport_security_state_static.json`; `https://www.cloudflare.com/tld-policies/`; `https://developers.cloudflare.com/registrar/faq/`; `https://developers.cloudflare.com/registrar/registrar-api/`; `https://www.cloudflare.com/learning/dns/glossary/premium-domains/`; `https://api.porkbun.com/api/json/v3/pricing/get`; `https://porkbun.com/checkout/search`; `https://www.namecheap.com/domains/registration/results/`; `https://pypi.org/pypi/dew-ml/json`; `https://pypi.org/pypi/dew/json`; `https://api.github.com/repos/AshishKumar4/dew`; `https://registry.npmjs.org/dew`.
