"""
Teste completo de divisão de chunks e limpeza de texto
SEM carregar o modelo (apenas testa o preprocessamento)
"""
from chatterbox.text_preprocessor import normalize_text, split_into_chunks, estimate_generation
import json

# Texto do teste - história completa do Nordstream
TEXTO_TESTE = """September 26, 2022. Just before 8:00 p.m., a sailing ship is drifting gently on the Baltic Sea. The captain is getting ready for the night. He checks his radar and sees two red blips flashing on the screen. He stares out into the darkness. Nothing. So, he radios a nearby ship. Their crew reports something strange. giant whirlpools in the water. When his ship approaches the spot, the captain is hit by a sharp and sour smell. Gas. He immediately notifies the Swedish authorities. They confirm his suspicion. There's an underwater leak. Tons of natural gas are flowing uncontrollably into the Baltic Sea. Nordstream 1 and two are a twin set of undersea gas pipelines running from Russia to Europe. the longest in the world and they've been hit. Four separate explosions at two different locations roughly 50 miles apart. It soon becomes clear that this was not an accident. It was an attack. One of the largest acts of sabotage in history, a violent act that triggered the largest methane leak ever recorded and threatened the energy supply of Europe's largest economy. Enormous teams of journalists race to uncover who did this and why, while authorities stay suspiciously quiet. Out of the [ __ ] of war, an unbelievable story begins to take shape. One of conflicting intelligence, possible false flag operations, and a scramble for evidence amid Europe's largest armed conflict since World War II. A story of speed camera photo clues, an aging yard, traces of military grade explosives and CIA training, and of a rectac crew of patriots on a hailmary mission that would change the geopolitical landscape of an entire continent. Gas bubbling up over an area of more than one square kilometer in the Baltic Sea. A boiling cauldron. The busy Baltic Sea bursting with gas from ruptured Russian Nordstream reinforced pipelines. One of the investigators told me that this case is by far the most relevant political sensitive and important case in the history of the Federal Republic. There are no shortage of theories about who or what was behind the blast. These are Nordstream 1 and two, the two largest underwater pipelines in the world. Total cost about €17 billion. They stretch across the Baltic Sea, running straight from Russia to Germany. Back in September 2005, Vladimir Putin and then Chancellor of Germany Gada shake hands on the deal to build the first pipeline. It's Shrea's last big move before leaving office. Just weeks after losing reelection, Shrea lands himself a new gek as chairman of Nstream Ai. He becomes the top lobbyist for Russian energy giant Gasprom. His successor, Mer keeps pushing the project forward. By 2011, cheap Russian natural gas is flowing straight into Germany. Russia is making roughly€ 10 billion a year. Economically, Nordstream 1 is a huge success for both countries. Soon enough, Russia is supplying nearly half of Germany's natural gas, and talks for a second pipeline are already underway. But outside Germany, Nordstream sparks some serious backlash. The US warns the project could give Russia dangerous leverage, turning gas into a geopolitical weapon. And the European Union also wants to reduce its dependence on Russian gas. Eastern European countries are pushing back especially hard. Until now, Russia's only way to get gas into Europe was through Ukraine, Poland, and the Baltics, paying hefty transit fees along the way. The Nordstream pipelines allow Russia to go around these countries. Some military analysts think Putin might have invaded Ukraine much sooner, but he still needed his gas pipelines that run through Ukrainian territory. Either way, Berlin remains undeterred by the criticism. Russia has already illegally annexed Crimea when Germany signs off on the second pipeline. In late 2021, Nordstream 2 is ready to go. But when Russia's full-scale invasion of Ukraine becomes imminent, Germany pulls the pluck on Nordstream 2. A few months later, Putin cuts off gas through Nordstream 1. Both pipelines stay loaded with gas, though, ready to reopen at any moment. And then in September 2022, this [Music] happens. So, who's behind it? This could just as well be the plot of a gripping who done it. Some stories and images are so striking they end up decorating our walls. But let's be real, traditional posters tend to be like Russian pipelines. They rip. This plate is different. They offer highquality metal prints, bold, durable, and easy to set up. You just stick a magnet to your wall, snap the artwork on, and it's done. No tools, no holes, no hassle. You can choose from over 2 million beautiful designs from abstract landscapes to officially licensed stuff. Think iconic worlds and franchises like Star Wars, The Witcher, or Cyberpunk 2077, a personal favorite we actually got on our walls. The new texture series looks especially good with those raised textures and 3D contours. Check out these incredible NASA designs, for example. If you ever feel like switching things up, they swap out in seconds. And now's the time to get yours. This plate is turning 12 and they're throwing their biggest sale of the year. 33% off one to two disc plates, 37% off three, and 42% off four or more. Every order comes with surprise gifts. And yes, the discount works on Textra, too. Just use the code Fern or click the link in the description. /adfern. The Baltic Sea is one of the most closely monitored bodies of water in the world. Larger ships have to constantly share their GPS location, and the kind of explosives you'd need to take out two pipelines at once is extremely hard to come by. Investigators soon agree whoever pulled this off had access to serious resources. This must have been a state actor. The first prime suspect is Russia. There's also a possible financial motive. Russian state-owned energy giant Gasprom faced massive damage claims after cutting off gas supplies in summer 2022. Blowing up the pipelines could give them a legal excuse to duck out of those contracts. This is Hgashtak, the deputy editor-inchief at German weekly newspaper Dite. He has spent months digging into the case. Uh when I first heard about the explosions, uh I immediately thought it must be the Russians. It just fits their classic pattern of obstruction, of sabotage, of provocation. And to me, it just made perfect sense. And initially the suspicion seems to hold up. 4 days before the explosion, a Danish naval vessel patrols near the island of Bourneholm. It spots six Russian ships right where the pipelines will be blown apart 4 days later. They're dark ships, meaning their AIS transponders are switched off and they're not broadcasting their location. One of them is the SS750, a 100yard special ops vessel designed for underwater missions. On board, a mini submarine equipped with cranes for heavyduty lifting. Exactly the kind of setup you'd need to plant explosives on a pipeline. 3 months earlier, two other Russian military vessels have been spotted dark shipping the very same waters right around the future blast site. What exactly were they doing there? Around the same time, suspicion also turns toward the United States. Just weeks before Russia invaded Ukraine, President Biden and German Chancellor Schultz held a joint press conference. If Russia invades, there will be no longer a Nordstream 2. We we will bring an end to it. 6 months later, both pipelines explode. That's when Seamour Hirs steps in. a Pulitzer Prizewinning investigative journalist known for uncovering major scandals, but also for spreading highly controversial theories. He claims that US Navy divers planted the explosives on the pipelines during a NATO exercise in summer 2022. He also says they got help from a Norwegian Navy mind sweeper and surveillance aircraft. The alleged motive to push Russian gas out of the European market and make room for American liqufied natural gas. Norway is another key supplier and would benefit too. But Hersh's story quickly comes under fire. The flight data recordings from Norway do not match his timeline. And the ships he mentions haven't even been near the site. My impression is that Sai Hersh had just one source, an old source from the world of intelligence. Someone who picked up fragments of a story and then filled in the rest. Hersh admits that he had only this one source and that this source didn't witness anything firsthand. And to me, that's already a red flag. Hirsh is 88 by now. In recent years, he has been repeatedly criticized for relying on thin sources and pushing shaky claims. He no longer comments publicly on his Nordstream story. The strongest lead points to neither Russia nor the United States. Shortly after the explosion, Hashtag gets a crucial tip off from a well-connected source inside the intelligence community. So, I called someone I've known for a long time and asked, "Was it Russia?" He just looked at me and with a very long and quiet and doubtful look, he slowly shook his head. [Music] After the explosions, German authorities immediately launch an investigation. The Federal Prosecutor's Office picks up the case. The case is so politically sensitive that several major media outlets start joining forces. You'll find the links to their impressive research in the description. Julia Klaus is an investigative journalist at public German TV broadcaster set. She is part of one of the extensive research teams. She tells us that despite the heavy secrecy, they were able to get their hands on one crucial piece of information. So, we picked up that the Andromeda yacht may have played a role in the operation. The Andromeda is a small German sailing yacht just over 15 yd long. But could such a small boat be used to pull off such a sophisticated militarygrade underwater demolition? It does have one major advantage. It's discreet. And unlike larger ships, it doesn't have to broadcast its position. The journalists uncovered that the Andromeda was chartered between August 27 and September 24 from a German boat rental company called Mola Yachting. The total rent of €11,900 was wired by a Polish travel agency. They submitted two passport copies, one Bulgarian, one Romanian. The name on the Romanian ID was Stefan Maku. Maraku lives in Muldova, a small landlocked country in Eastern Europe. Julia Klaus and her team get on a plane and pay him a visit. We were sitting in his garden and when we showed him uh the passport number, he immediately recognized it. He said it matched his old passport, which had long been expired. Uh, but he had like no idea how his personal data ended up in connection with the Nordstream bombing. The passport was fake, just like the Bulgarian one issued to a man named Mika Popov. Another team of journalists head out to V Harbor on the German Baltic island of Rugen, where the Andromeda crew has last been seen. They speak to the harbor master who remembers seeing them. They talk to other witnesses and even rent the Andromeda themselves. Step by step, they are able to piece together the route the boat most likely took. Meanwhile, Stark and his team follow up on another lead, the Polish travel agency who paid for the yacht. It turns out to be a shell company. Behind that shell company, we found a Ukrainian businessman, someone who was operating in the gray zones of uh the underground economy in Kiev. He clearly was being used obviously as a favor to someone higher up in the chain of command. Charter that boat, pay the bill, don't ask questions. In March 2023, Stark and his research team published their findings. The timing is delicate. The German public is locked in a heated debate. should the country send tanks to support Ukraine. And now reports suggest that Ukrainians may have attacked a German Russian energy pipeline. That's politically explosive news. There's also speculation that the Andromeda lead might be a trap, a false flag, something deliberately planned by Russia to confuse the public and stir political doubt in Germany. But behind closed doors, senior officials already know who they should be looking at. German authorities have been zeroing in on a group that likely pulled the strings. June 2022, 3 months before the Nordstream blast, the German Foreign Intelligence Service BND receives a top secret report. It's from the Dutch military intelligence service MIV. They say they've been tipped off about an imminent sabotage operation targeting Nstream 1. The suspected culprits, a Ukrainian special forces unit traveling with fake passports and a rented boat. The attack is set to be scheduled for June 19. That's only a few days away. Shortly after, another warning comes in from the CIA. They too have been alerted by the Dutch. But June 19 comes around and nothing happens. The warnings are written off as false alarms. No one notifies the German Federal Police or the German Navy or counterterrorism task forces. 3 months later, the pipelines do get hit almost exactly the way the Dutch report had predicted. So, right away, investigators turned their attention toward Ukraine. January 2023, 4 months after the explosions, on board the Andromeda, investigators find compromising material, DNA, fingerprints, a water bottle, and a single barefoot shoe. And on a table and in the toilet, traces of a powerful underwater explosive called HMX, also known as Octogen. That's the exact same compound found at the Nordstream blast sites. At this point, the investigators are sure this boat was part of the secret sabotage operation. But who did it? Then comes the key piece of evidence, a photo. It's early morning. A van beats down a rural road. The driver has been hired to take a group of Ukrainians to the port. He has no idea who they really are, but he carries their diving gear in their back. He is going fast. too fast. The speed camera photo shows a white Citroen space tour with Ukrainian license plates. Next to the driver sits an unidentified passenger. Investigators track down the clueless driver. He confirms the identity of the passenger, a Ukrainian national named Wadomia Z. Bingo. The grainy snapshot marks a turning point. In June 2024, German investigators issue a European arrest warrant for him. Over time, more and more connections start pointing towards Ukraine. Investigators managed to fully identify the alleged Andromeda crew. Journalists even speak with a Ukrainian informant who claims he helped them plan the attack. Today, authorities have a fairly clear picture of what most likely happened. [Music] Spring 2022, a rented office in Kiev, Ukraine. Inside as a small commando team, allegedly led by a man named Roman Sherinsky. He appears to be a legend in Ukrainian intelligence circles. Several members of his unit have also trained with the CIA. Their specialty covert ops and sabotage. They are Ukrainian patriots. And after Russia's full-scale attack on their country, they're eager to strike back hard. They choose to target the Nordstream pipelines, a symbol of Russian power that's been pumping billions into Putin's war chest. They need trained divers to pull off such a deep water operation. They short list 20 candidates and end up picking four, all civilians. Two soldiers complete the team. A mly crew of six Ukrainians who barely know each other. [Music] September 8th, 2022. It's late morning at V Harbor on Rug Island. A saleard pulls into the dock behind the harbor office. It's hailing straight from Rosto, where it was chartered 2 days ago by the small crew, posing as tourists. But they're not here for a sailing trip. They've come to carry out one of the most brazen acts of sabotage in history. Introducing Volto Zi, a heating engineer in his mid-40s who lives in Poland. He's also a scuba instructor specialized in deep diving expeditions. Next up is Daria Kmanova. At least that's her cover name, early 40s, petite, brown hair. She and her husband run a small diving school in Ukraine. Vote Z used to work for them as a scuba instructor. Then there's Daria's husband, cenamed Folgot, another experienced deep sea diver, pale, boyish, thinning hair. The fourth member is the commander, codein named Rubik, seasoned combat soldier, built like a tank and a capable diver himself. He's the mission lead aboard Andromeda. He stays in constant contact with Kief via 16 burner phones and a stash of prepaid Sims. The fifth one is the captain, an active duty soldier, grumpy, hostile, broadframe. A witness will later describe him as a guy who's had one too many hamburgers. The sixth one is the assistant, short, middle-aged. The group gets to work right away. A harbor worker watches them from a distance. They're unloading grocery bags from a van, carrying them to the boat across the parking lot. Each walk is about a 100 yards. The assistant is standing next to an unusual amount of compressed air cylinders. The harbor master walks over. You guys need a hand? You can pull the van closer if you want. The assistant shakes his head. We're good. The harbor master doesn't know that these cylinders are not for diving. Each one of them is packed with up to 35 kilos of HMX explosive. This would be the last real chance to stop them. It passes shortly after the Andromeda casts off. Operation Diameter is underway. Here's the path both Nordstream pipelines take. Nordstream 1 and the recently completed Nordstream 2. From Rugan, the Andromeda makes its way to Bourneholm and docks again on the small Danish island of Christian. Out on the open water, the crew starts assessing the situation. The boat is past its prime. The wood caks with every turn. The walls are scuffed and the autopilot's broken. The 75 horsepower diesel engine rattles like an old farm tractor. It's quiet out here on the Baltic Sea. They only pass a few ships and no one really pays attention to a small sailboat bobbing on the waves. The crew uses sonar technology to echolocate the pipelines. They probably conduct multiple expedition dives, but progress is slow. They can only do one dive per day. By the end of the second week of September, a storm rolls in. Winds of 20 knots whip across the sea, and the waves climb up to nearly 7 ft. The Andromeda takes shelter in a small harbor in Sandham, Sweden. A few days later, the sky is clear and the wind dies down, but the waves are still rocking the boat and most of the crew have fallen seasick. Conditions are barely stable enough to dive. The sonar hums as the crew awaits intention. Then, finally, a signal. They found the pipeline about 80 yards below, directly beneath the Andromeda. It's go time. Now or never. They lower a weighted rope into the water. About a minute later, it hits the seabed. A plastic canister wrapped in a trash bag bobs on the surface as a makeshift boy. It's marking the spot so they don't drift off. Two divers strap on their scuba tanks. Then they grab a separate tank filled with high-powered explosives. They carefully slip off the stern and attach the tank to the rope. One last check and they disappear. Following the rope down into the dark, this is the pipeline 260 ft below the surface. It's cold and dark. The pressure on the human body down here is about nine times higher than at sea level. Two beams of light cut through the darkness. They only have about 15 minutes on the seabed. To plant the bomb, they need to locate a connection joint along the pipeline. Their headlights can only illuminate 6 to 7 ft ahead of them. After a bit, they find a seam. They place a silicone pad between the bomb and the pipe to amplify the blast. Done. All set. At this depth, standard scuba gear won't cut it. The divers are most likely using advanced equipment, rebreathers, multiple tanks, and try mix breathing gases. To avoid decompression sickness, they need to ascend slowly with several stops along the way. The whole process takes at least 40 minutes. The Andromeda has kept moving during the dive to avoid suspicion. Now, the ship returns to the makeshift buoy. The crew is relieved. They carry out multiple additional dives over the course of several days. They plant at least six bombs across the two pipelines. Each charge is fitter with a timer set to go off on September 26. On the morning of September 23rd, the Andromeda is spotted drifting at Rastto Harbor. The staff from Mola Yading arrive to check in the rental boat, but find it empty. The crew is already gone. Unexplained leaks in two major pipelines from Russia into Europe have become a major cause of concern for the European Union. Denmark, Poland, and Sweden say they believe leaks in two major Russian gas pipelines to Europe are the result of sabotage. Tonight, sabotage at sea. That's what President Biden is calling the leaks and explosions on the Nordstream pipelines. 3 days later, the sabotur can watch the outcome of their mission live on TV. It worked except for one detail. One of the two lines of Nordstream 2 remains intact. The crew accidentally hit the first line [Music] twice. June 2024, Polish authorities receive a European arrest warrant for Vadim Zi by Germany's federal prosecutor's office. Inside they find the suspect's listed address in a residential suburb of Warsaw. Following EU law, Polish prosecutors are expected to act on such a warrant and detain a suspect. No questions asked, but apparently the polls do nothing. It's very likely the Poles who tipped off uh the Ukrainian diver and also the Ukrainian government in order to get him uh out of the country in time before he could have been arrested and uh put on trial in Germany. On the morning of July 6th, at exactly 6:20 a.m., a BMW with Ukrainian diplomatic plates crosses the Polish border into Ukraine. The co-driver is Vladimir Z. The suspected Sabotur has vanished ever since. And for now, no arrest warrants have been issued for the rest of the crew due to a lack of evidence. To this day, it is not definitely proven who was behind the Nordstream attack. All suspects deny any involvement. That said, there's a growing body of evidence pointing toward the Andromeda theory. The yacht, the forge documents, the diving gear, the witness testimonies, the DNA. It all paints a compelling picture. There are credible indications that the former commanderin-chief of Ukraine's armed forces approved the plan. Still, nothing is off the table. Some experts haven't ruled out the possibility that the Andromeda may have been part of a more elaborate deception or false flag operation by Russia or the US, but many experts do believe the attack was likely carried out by Ukrainian patriots. Whether it happened with the government's blessing is up for debate. You could argue that the Ukrainian state wouldn't want to provoke a key weapon supplier like Germany and risk losing critical military support. The alleged attackers must have seen the pipelines as an immediate threat to Ukraine, as a tool of Russian leverage and thus a legitimate target. Today, no Russian gas flows through Nordstream 102. Germany has managed to break free of its dependence faster than expected. Nordstream was the most expensive, most ambitious, and most controversial underwater pipeline project the world has ever seen. Now, both pipelines sit idle, partially destroyed, deep beneath the Baltic Sea. Since March 2025, there's been talk of bringing Nordstream 2 back online. A US investor has reportedly expressed interest in buying and repairing the pipeline, but no one knows just how realistic that is, or how Germany and the rest of Europe might respond."""


def analise_completa():
    print("=" * 80)
    print("TESTE COMPLETO DE DIVISÃO DE CHUNKS E LIMPEZA DE TEXTO")
    print("=" * 80)
    print()
    
    # 1. ANÁLISE DO TEXTO ORIGINAL
    print("📊 ANÁLISE DO TEXTO ORIGINAL")
    print("-" * 80)
    print(f"Tamanho total: {len(TEXTO_TESTE)} caracteres")
    print(f"Palavras aproximadas: {len(TEXTO_TESTE.split())}")
    print(f"Linhas: {TEXTO_TESTE.count(chr(10)) + 1}")
    print()
    
    # 2. TESTE DE NORMALIZAÇÃO
    print("🧹 TESTE DE NORMALIZAÇÃO (normalize_text)")
    print("-" * 80)
    texto_normalizado = normalize_text(TEXTO_TESTE)
    print(f"Texto original tinha: {len(TEXTO_TESTE)} caracteres")
    print(f"Texto normalizado tem: {len(texto_normalizado)} caracteres")
    print(f"Redução: {len(TEXTO_TESTE) - len(texto_normalizado)} caracteres")
    print()
    
    # Mostrar exemplo de limpeza
    print("Exemplo de limpeza:")
    amostra_antes = "Out of the [ __ ] of war... [Music] happens."
    amostra_depois = normalize_text(amostra_antes)
    print(f"  ANTES:  '{amostra_antes}'")
    print(f"  DEPOIS: '{amostra_depois}'")
    print()
    
    # 3. TESTE DE CHUNKING COM DIFERENTES TAMANHOS
    print("✂️ TESTE DE CHUNKING COM DIFERENTES TAMANHOS")
    print("-" * 80)
    
    tamanhos_teste = [150, 200, 250, 300, 400]
    
    for max_chars in tamanhos_teste:
        print(f"\n--- Max chars: {max_chars} ---")
        chunks = split_into_chunks(TEXTO_TESTE, max_chars=max_chars)
        stats = estimate_generation(chunks)
        
        print(f"  Total de chunks: {stats['num_chunks']}")
        print(f"  Tamanho médio: {stats['avg_chars']} chars/chunk")
        print(f"  Tempo estimado: {stats['estimated_minutes']} minutos")
        print(f"  Caracteres totais: {stats['total_chars']}")
        
        # Análise de distribuição
        tamanhos = [len(c) for c in chunks]
        print(f"  Menor chunk: {min(tamanhos)} chars")
        print(f"  Maior chunk: {max(tamanhos)} chars")
        print(f"  Chunks acima do limite: {sum(1 for t in tamanhos if t > max_chars)}")
    
    # 4. ANÁLISE DETALHADA COM max_chars=250 (configuração padrão)
    print("\n" + "=" * 80)
    print("📋 ANÁLISE DETALHADA (max_chars=250 - configuração padrão)")
    print("=" * 80)
    
    chunks = split_into_chunks(TEXTO_TESTE, max_chars=250)
    
    print(f"\nTotal de chunks gerados: {len(chunks)}")
    print("\n--- PRIMEIROS 5 CHUNKS ---")
    for i, chunk in enumerate(chunks[:5], 1):
        print(f"\nChunk #{i} ({len(chunk)} chars):")
        print(f"  '{chunk[:80]}{'...' if len(chunk) > 80 else ''}'")
    
    print("\n--- ÚLTIMOS 3 CHUNKS ---")
    for i, chunk in enumerate(chunks[-3:], len(chunks)-2):
        print(f"\nChunk #{i} ({len(chunk)} chars):")
        print(f"  '{chunk[:80]}{'...' if len(chunk) > 80 else ''}'")
    
    # 5. ANÁLISE DE QUALIDADE DOS CHUNKS
    print("\n" + "=" * 80)
    print("✅ ANÁLISE DE QUALIDADE DOS CHUNKS")
    print("=" * 80)
    
    chunks_quebrados = []
    chunks_muito_pequenos = []
    chunks_ideais = []
    
    for i, chunk in enumerate(chunks, 1):
        tamanho = len(chunk)
        
        # Verifica se termina com pontuação adequada
        termina_bem = chunk[-1] in '.!?'
        
        if tamanho < 100:
            chunks_muito_pequenos.append((i, tamanho, chunk))
        elif tamanho > 250:
            chunks_quebrados.append((i, tamanho, chunk))
        elif termina_bem:
            chunks_ideais.append((i, tamanho))
    
    print(f"\n✅ Chunks ideais (100-250 chars, termina com pontuação): {len(chunks_ideais)}")
    print(f"⚠️  Chunks muito pequenos (<100 chars): {len(chunks_muito_pequenos)}")
    print(f"❌ Chunks acima do limite (>250 chars): {len(chunks_quebrados)}")
    
    if chunks_muito_pequenos:
        print("\n--- Chunks muito pequenos ---")
        for i, tam, texto in chunks_muito_pequenos[:3]:
            print(f"  Chunk #{i} ({tam} chars): '{texto}'")
    
    if chunks_quebrados:
        print("\n--- Chunks acima do limite ---")
        for i, tam, texto in chunks_quebrados[:3]:
            print(f"  Chunk #{i} ({tam} chars): '{texto[:100]}...'")
    
    # 6. SALVAR RESULTADO COMPLETO EM JSON
    print("\n" + "=" * 80)
    print("💾 SALVANDO RESULTADOS EM JSON")
    print("=" * 80)
    
    resultado = {
        "texto_original": {
            "tamanho": len(TEXTO_TESTE),
            "palavras": len(TEXTO_TESTE.split())
        },
        "texto_normalizado": {
            "tamanho": len(texto_normalizado),
            "reducao": len(TEXTO_TESTE) - len(texto_normalizado)
        },
        "chunks": {
            "total": len(chunks),
            "configuracao": {
                "max_chars": 250
            },
            "estatisticas": {
                "tempo_estimado_minutos": estimate_generation(chunks)['estimated_minutes'],
                "tamanho_medio": estimate_generation(chunks)['avg_chars'],
                "menor": min(len(c) for c in chunks),
                "maior": max(len(c) for c in chunks)
            },
            "qualidade": {
                "ideais": len(chunks_ideais),
                "muito_pequenos": len(chunks_muito_pequenos),
                "acima_limite": len(chunks_quebrados)
            },
            "lista_completa": [
                {
                    "numero": i,
                    "tamanho": len(chunk),
                    "texto": chunk
                }
                for i, chunk in enumerate(chunks, 1)
            ]
        }
    }
    
    with open("resultado_teste_chunks.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    
    print("✅ Arquivo salvo: resultado_teste_chunks.json")
    print()
    
    # RESUMO FINAL
    print("=" * 80)
    print("🎯 RESUMO FINAL")
    print("=" * 80)
    print(f"✅ Texto normalizado com sucesso ({len(TEXTO_TESTE) - len(texto_normalizado)} chars removidos)")
    print(f"✅ {len(chunks)} chunks gerados")
    print(f"✅ Tempo estimado de narração: {estimate_generation(chunks)['estimated_minutes']} minutos")
    print(f"✅ Qualidade: {len(chunks_ideais)}/{len(chunks)} chunks ideais ({len(chunks_ideais)/len(chunks)*100:.1f}%)")
    
    if len(chunks_quebrados) == 0:
        print("✅ PERFEITO: Nenhum chunk ultrapassa o limite de 250 caracteres!")
    else:
        print(f"⚠️  ATENÇÃO: {len(chunks_quebrados)} chunks ultrapassam o limite")
    
    print("\n✅ Teste completo finalizado!")
    print("=" * 80)


if __name__ == "__main__":
    analise_completa()
