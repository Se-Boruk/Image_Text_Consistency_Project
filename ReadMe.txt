#Projekt
###############################
#Submission_system_repo

#Testowac modele za pomoca submission_evaluator.py (symuluje co bedzie sie dzialo bo submission na stronke kapusty)

# Submission_template zawiera instrukcje jak skonstruowac .zip folder gotowy do wrzucenia


###########################################################
#Kroki:
###########################################################

###################################
#1)
Pierwszy krok to pobranie bazy danych:
Pobrane bazy danych to: 
- coco (ogolne obrazy)
- flickr30k (codzienne sytuacje, wydarzenia i ludz)
- ade20k (sceny, natura itp.)

Nastepnie zostaly im przypisane opisy (captions) z bazy danych localized narratives
Domyslnie jest ona stworzona dla szczegolowych opisow powiazanych z miejscem na obrazie ale nada sie rowniez bardzo dobrze jako opis
calego obrazu ze wzgledu na jej bardzo szczegolowe opisy

https://google.github.io/localized-narratives/

Calosc zostala zapisana w formacie arrow. Baza danych zawiera szczegolowy opis obrazu oraz sam obraz. Sa to tylko przypadki pozytywne (tj. obraz z przypisanym poprawnym opisem)
###################################
#2)




#Prepared codes:
#############################################
#Tokenizer_lib   	X
#Analyze_logs    	X
#Database_functions 	Half
#Architectures		X
#Config			Adjust the hashes or just leave them
#Evaluate test		Not done
#Main_train		Half - main train loop to adjust
#Functions		To adjust
#Database_create_v3	To adjust ALL


Submission 1: 
PRóba 1

Submission 2:
Dałem po prostu zwykły model który był z epoki 25 a nie best (52)

Submission 3:
Zmiany:
-lematyzacja tekstu
- zwiększony sequence length do 128, i min frequency do 2 (po lematyzacji jest nawet mniejszy słownik)
- Zmniejszony nacisk na visual img loss, mniejsza augmentacja i mniejsza waga straty










