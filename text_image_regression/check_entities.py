# -*- coding: utf-8 -*-

from deeppavlov import configs, build_model

ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)
print(ner_model([
"Meteorologist Lachlan Stone said the snowfall in Queensland was an unusual occurrence \
  in a state with a sub-tropical to tropical climate.",
"Церемония награждения пройдет 27 октября в развлекательном комплексе Hollywood and \
  Highland Center в Лос-Анджелесе (штат Калифорния, США).", 
"Das Orchester der Philharmonie Poznań widmet sich jetzt bereits zum zweiten \
  Mal der Musik dieses aus Deutschland vertriebenen Komponisten. Waghalter \
  stammte aus einer jüdischen Warschauer Familie."]))