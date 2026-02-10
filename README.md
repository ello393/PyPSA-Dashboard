# PyPSA-Dashboard
plotly-based dashboard for interactive visualization of PyPSA-results

- Export aus PyPSA als netCDF (.nc)
- Dateibenennung bei Sensitivitäten oder Variantenvergleich so, dass aus der Be-nennung die Unterscheidung hervorgeht (z. B. Dateina-me_Strompreis_36ct_kWh)
- Bei Verwendung von Storage_units muss „max_hours“ gesetzt werden, damit die Kapazität im Dashboard korrekt berechnet wird.
- Für multi_investment_periods = True müssen Komponenten, die über mehrere Jahre optimiert werden, mit „Komponentenname_Baujahr“ indiziert werden.
- Carrier müssen für alle Komponenten außer Links/ Lines definiert werden. Das Format dabei soll z. B. "Sektor_Sektor_Präzisierung" sein (Beispiel: "Strom_Strom_Netzbezug" oder "Wärme_Wärme_Erdgas"). 
- Bei nicht näher definierbaren Energieträgern wird "variabel" eingesetzt, für den Sektor sind "Strom" oder "Wärme" die ausschlaggebenden Keywords zur Zuordnung der Komponenten. Andere Sektorenbezeich-nungen werden automatisch in die Kategorie "Sonstige" einsortiert. 
- Vor dem ersten Unterstrich wird der Sektor ausgelesen, danach der Energieträger für die Legenden der Diagramme und eine mögliche Prä-zisierung des Energieträgers. Auf diese Weise bleibt sowohl die Möglich-keit einer Sektorenzuordnung als auch die einer CO2-Constraint vorhan-den.
- Einspeisung muss als Generator über sign = -1 implementiert werden. Der Komponentenname muss "Einspeisung" oder "einspeisung" enthalten, um im Sankey-Diagramm korrekt zugeordnet werden zu können.
- Store-Komponenten müssen immer über einen separaten Bus mit zwei Links (Laden/ Entladen) implementiert werden
- Alle Komponenten, die in der Wirtschaftlichkeitsberechnung berücksichtigt werden sollen, müssen Lifetime hinterlegt haben (für Annualisierung)
- Im Variantenvergleich dürfen nur MIP mit MIP und Single Year mit Single Year verglichen werden
