#### Leon Henne, 22.02.2022
# Data Exploration Project
## Predicting the survivability of the Titanic passengers

### Erläuterung und Vorstellung des Datensatzes
Der in diesem Projekt untersuchte Datensatz basiert auf den Passagierdaten der am 15. April 1912 versunkenen Titanic. Zu 891 Passagieren sind Informationen bezüglich deren Person, wie Name, Geschlecht, Alter und Familienstatus, sowie deren Aufenthalt, z.B. Passagiernummer, Schiffsklasse, Ticket, Preis, Einstiegsort, Kabinennummer und schließlich ihres Überlebensstatus bekannt.

### Datenvorverarbeitung
Dropping ["Cabin", "Embarked", "Ticket","Fare", "Name", "PassengerId"]
Replacing missing age values with the average (median) age
### Feature Engineering
Sex Spalte in 0 und 1 um sie zu Integern zu konvertieren und in einen vergleichbaren Raum zu projezieren
Age Spalte in Range zwischen 0 und 1 bringen ? ( / 100 )
Summe der Angehörigen berechnen (Parch + SibSp) ? in Bereich zwischen 0 und 1 bringen ? 
### Teilen in Trainings- und Testdaten

### Erläuterung des Machine Learning Algorithmus

### Auswahl der Metriken

### Traingsergebnisse des Algorithmus

### Hyperparmametertuning

### Evaluation auf den Testdaten

### Schwachstellen und Verbesserungsmöglichkeiten im Algorithmus