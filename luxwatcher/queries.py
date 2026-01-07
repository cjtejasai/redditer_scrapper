from __future__ import annotations


def get_queries() -> list[str]:
    # Keep queries explicit and simple (Firecrawl already supports `tbs=qdr:d` for last 24h).
    # Real estate (from your notebook), plus additional luxury signals:
    # - Luxury cars / supercars
    # - Luxury watches
    # - Yachts / yacht charters
    # - Extra high-luxury signal: private jets / jet charter

    real_estate = [
        # Dubai — buyer intent
        "site:reddit.com/r/dubai looking to buy apartment dubai budget",
        "site:reddit.com/r/dubai looking to buy villa dubai family",
        "site:reddit.com/r/dubai moving to dubai and want to buy property",
        "site:reddit.com/r/dubai planning to buy property in dubai",
        "site:reddit.com/r/dubai best area to buy apartment for living",
        "site:reddit.com/r/dubai buy apartment to live not invest",
        "site:reddit.com/r/dubai mortgage approved buying apartment",
        "site:reddit.com/r/dubai off plan vs ready property buying",
        "site:reddit.com/r/dubai expat buying property advice",
        "site:reddit.com/r/dubai buying property golden visa",
        "site:reddit.com/r/dubai buy apartment dubai marina",
        "site:reddit.com/r/dubai buy apartment downtown dubai",
        "site:reddit.com/r/dubai buy apartment business bay",
        # Dubai — seller / broker intent
        "site:reddit.com/r/dubai selling apartment dubai",
        "site:reddit.com/r/dubai selling villa dubai",
        "site:reddit.com/r/dubai listing my property for sale",
        "site:reddit.com/r/dubai need help selling apartment",
        "site:reddit.com/r/dubai recommend real estate broker",
        "site:reddit.com/r/dubai recommend real estate agent",
        "site:reddit.com/r/dubai best real estate agent dubai",
        # Qatar — buyer intent
        "site:reddit.com/r/qatar looking to buy apartment qatar",
        "site:reddit.com/r/qatar planning to buy property in qatar",
        "site:reddit.com/r/qatar moving to qatar buy property",
        "site:reddit.com/r/qatar buy apartment the pearl",
        "site:reddit.com/r/qatar buy apartment lusail",
        "site:reddit.com/r/qatar expat buying property",
        # Qatar — seller / broker intent
        "site:reddit.com/r/qatar selling apartment qatar",
        "site:reddit.com/r/qatar need real estate agent",
        "site:reddit.com/r/qatar recommend property broker",
        # Spain — buyer intent (expats / luxury)
        "site:reddit.com buying property in spain advice",
        "site:reddit.com looking to buy apartment in spain",
        "site:reddit.com moving to spain buy property",
        "site:reddit.com expat buying property in spain",
        "site:reddit.com buy villa marbella",
        "site:reddit.com buy property costa del sol",
        "site:reddit.com buy apartment barcelona",
        "site:reddit.com buy apartment madrid",
        # Spain — seller / broker intent
        "site:reddit.com selling property in spain",
        "site:reddit.com need real estate agent spain",
        "site:reddit.com recommend real estate agent spain",
    ]

    # Luxury cars / supercars (different places + relevant subreddits)
    luxury_cars: list[str] = []
    # luxury_cars = [
    #     "site:reddit.com/r/dubai supercar rental recommendation",
    #     "site:reddit.com/r/dubai where to buy ferrari used",
    #     "site:reddit.com/r/dubai lamborghini dealer recommendation",
    #     "site:reddit.com/r/dubai import car from europe luxury",
    #     "site:reddit.com/r/abudhabi supercar dealer recommendation",
    #     "site:reddit.com/r/qatar supercar rental doha recommendation",
    #     "site:reddit.com/r/qatar where to buy luxury car",
    #     "site:reddit.com/r/saudiarabia luxury car dealer recommendation",
    #     "site:reddit.com/r/london luxury car dealer recommendation",
    #     "site:reddit.com/r/miami supercar rental recommendation",
    #     "site:reddit.com/r/LosAngeles luxury car dealer recommendation",
    #     "site:reddit.com/r/nyc luxury car lease recommendation",
    #     "site:reddit.com/r/monaco supercar dealer recommendation",
    #     "site:reddit.com/r/ferrari looking to buy advice",
    #     "site:reddit.com/r/lamborghini looking to buy advice",
    #     "site:reddit.com/r/rollsroyce looking to buy advice",
    # ]

    # Luxury watches
    luxury_watches: list[str] = []
    # luxury_watches = [
    #     "site:reddit.com/r/dubai rolex dealer recommendation",
    #     "site:reddit.com/r/dubai patek philippe where to buy",
    #     "site:reddit.com/r/dubai audemars piguet where to buy",
    #     "site:reddit.com/r/qatar rolex dealer doha recommendation",
    #     "site:reddit.com/r/london rolex AD waitlist advice",
    #     "site:reddit.com/r/rolex looking to buy first rolex",
    #     "site:reddit.com/r/Watches looking to buy patek",
    #     "site:reddit.com/r/Watches looking to buy richard mille",
    #     "site:reddit.com grey market watch dealer recommendation",
    #     "site:reddit.com watch dealer recommendation dubai",
    # ]

    # Yachts / superyachts
    yachts: list[str] = []
    # yachts = [
    #     "site:reddit.com/r/dubai yacht charter recommendation",
    #     "site:reddit.com/r/dubai looking to buy yacht",
    #     "site:reddit.com/r/dubai marina berth for yacht",
    #     "site:reddit.com/r/miami yacht charter recommendation",
    #     "site:reddit.com/r/miami looking to buy yacht",
    #     "site:reddit.com/r/spain yacht charter costa del sol recommendation",
    #     "site:reddit.com superyacht charter recommendation",
    #     "site:reddit.com looking to buy yacht advice",
    # ]

    # Extra high-luxury signal: private jets / jet charter
    private_jets: list[str] = []
    # private_jets = [
    #     "site:reddit.com private jet charter recommendation",
    #     "site:reddit.com gulfstream charter recommendation",
    #     "site:reddit.com netjets vs wheels up recommendation",
    #     "site:reddit.com looking to buy private jet",
    #     "site:reddit.com fractional jet ownership advice",
    # ]

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for q in (real_estate + luxury_cars + luxury_watches + yachts + private_jets):
        q = (q or "").strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out

