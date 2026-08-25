### nanogpt training records

bigram model, with a single embedding layer and no self-attention blocks.
```
step 0: train loss 4.7305, val loss 4.7241
step 300: train loss 2.8110, val loss 2.8249
step 600: train loss 2.5434, val loss 2.5682
step 900: train loss 2.4932, val loss 2.5088
step 1200: train loss 2.4863, val loss 2.5035
step 1500: train loss 2.4665, val loss 2.4921
step 1800: train loss 2.4683, val loss 2.4936
step 2100: train loss 2.4696, val loss 2.4846
step 2400: train loss 2.4638, val loss 2.4879
step 2700: train loss 2.4738, val loss 2.4911



CEThik brid owindakis b, bth

HAPet bobe d e.
S:
O:3 my d?
LUCous:
Wanthar u qur, t.
War dXENDoate awice my.

Hastarom oroup
Yowhthetof isth ble mil ndill, ath iree sengmin lat Heriliovets, and Win nghir.
Swanousel lind me l.
HAshe ce hiry:
Supr aisspllw y.
Hentofu n Boopetelaves
MPOLI s, d mothakleo Windo whth eisbyo the m dourive we higend t so mower; te

AN ad nterupt f s ar igr t m:

Thin maleronth,
Mad
RD:

WISo myrangoube!
KENob&y, wardsal thes ghesthinin couk ay aney IOUSts I&fr y ce.
J
```

this time, position embedding was added and the generation logic was changed.
and an extra linear layer.

```
step 0: train loss 4.4801, val loss 4.4801
step 300: train loss 2.5404, val loss 2.5566
step 600: train loss 2.5160, val loss 2.5335
step 900: train loss 2.4967, val loss 2.5149
step 1200: train loss 2.5106, val loss 2.5254
step 1500: train loss 2.4853, val loss 2.5109
step 1800: train loss 2.4966, val loss 2.5198
step 2100: train loss 2.4949, val loss 2.5100
step 2400: train loss 2.4937, val loss 2.5102
step 2700: train loss 2.5040, val loss 2.5114



CExthantrid owindikis s, bll

HAPen bube t e.
S:
O:
IS:
Folatangs:
Wanthar u qurthe. bar dilasoate awice my.

Hastatom o mup
Yowhthatof isth ble mil; dilll,

W:

Ye s, hain latisttid ov ts, and Wh pomano.
Swanous l lind me l.
MIshe ce hiry ptupr aisspllw y. w'stoul noroopetelaves
Momy ll, d mothake o Windo wh t eiibys the m douris TENGByore s poo mo th; te

AN ad nthrupt f s ar irist m:

Thin maleronth, af Pre?

Whio myr f-
LI har,
S:


Thardsal this ghesthidin cour ay aney Iry ts I f my ce hy
```

adding a single self-attention block, and here is what i got:
```
E:\workspace\pyscripts\.venv\Scripts\python.exe E:\workspace\pyscripts\neural_networks\messygpt.py 
step 0: train loss 4.2000, val loss 4.2047
step 500: train loss 2.6911, val loss 2.7087
step 1000: train loss 2.5196, val loss 2.5303
step 1500: train loss 2.4775, val loss 2.4829
step 2000: train loss 2.4408, val loss 2.4523
step 2500: train loss 2.4272, val loss 2.4435
step 3000: train loss 2.4130, val loss 2.4327
step 3500: train loss 2.3956, val loss 2.4212
step 4000: train loss 2.4041, val loss 2.3992
step 4500: train loss 2.3980, val loss 2.4084

Whent iknt,
Thowi, ht son, bth

Hiset bobe ale.
S:
O-' st dalilanss:
Want he us he, vet?
Wedilas ate awice my.

HDET:
ANGo oug
Yowhavetof is he ot mil ndill, aes iree sen cie lat Herid ovets, and Win ngarigoerabous lelind peal.
-hule onchiry ptugr aiss hew ye wllinde norod atelaves
Momy yowod mothake ont-wou whth eiiby we ati dourive wee, ired thoouso er; th
To kad nteruptef so;
ARID Wam:
ENGCI inleront ffaf Pre?

Wh om.

He-
LIERCKENIGUICar adsal aces ard thinin cour ay aney Iry ts I fr af ve y
```

now a four heads multi head attention block was added.

```angular2html
step 0: train loss 4.2227, val loss 4.2226
step 500: train loss 2.6592, val loss 2.6733
step 1000: train loss 2.4980, val loss 2.5064
step 1500: train loss 2.4291, val loss 2.4349
step 2000: train loss 2.3716, val loss 2.3844
step 2500: train loss 2.3417, val loss 2.3561
step 3000: train loss 2.3149, val loss 2.3347
step 3500: train loss 2.2918, val loss 2.3171
step 4000: train loss 2.2895, val loss 2.2868
step 4500: train loss 2.2748, val loss 2.2858

Whent if bridcowd, whis byer that set bobe toe anthr-and mealleands:
Warth foulque, vet?
Wedtlay anes wice my.

HDY'n om oroug
Yowns, tof is heir thil; dill, aes isee sen cin lat Hetilrov the and Win now onderabousel.

SFAUS:
Shenser cechiry prugh aissthe, ye wing, u not
To thig I whomeny wod mothake ont---An hat evibys wietit, stile weeshirecs poor gier; to
To k danteref If sor; igre! mef thre inledo the af Pre?

WISo myay I sup!
Atied is:
Sadsal the E'd st hoin couk aar tey Iry to I frouf voul
```

FeedForward was added.

```
step 0: train loss 4.1996, val loss 4.1995
step 500: train loss 2.5993, val loss 2.6077
step 1000: train loss 2.4629, val loss 2.4651
step 1500: train loss 2.3974, val loss 2.3951
step 2000: train loss 2.3297, val loss 2.3470
step 2500: train loss 2.3018, val loss 2.3221
step 3000: train loss 2.2828, val loss 2.2936
step 3500: train loss 2.2495, val loss 2.2721
step 4000: train loss 2.2435, val loss 2.2468
step 4500: train loss 2.2286, val loss 2.2411

And the Rorincowf,
This by be mad thom obe to tarver-' my dall and bar hiphe us hat tot?
Wedtlacoate aw crup and not, ut onour
Yowns, tof it he cove lend lincath is ees, hain lat Het dulvets, and to poman is wables lill dite ullliser cecrivy prupt aiss hew youn's and knamopetell lownomthy wod moth keacal---A wher eiicks to thour rive cees ineds pood of he thu the hanterth fo so;; igis! my to thy ale ontat af Pried my of.
WHINY ICHARD:
Pois:
Ardsal the Eget to uin cour ay andy Rry to chan the!
An
```

residual connection.

```
step 0: train loss 4.6255, val loss 4.6233
step 500: train loss 2.3884, val loss 2.3848
step 1000: train loss 2.2707, val loss 2.2693
step 1500: train loss 2.1884, val loss 2.2097
step 2000: train loss 2.1471, val loss 2.1831
step 2500: train loss 2.1079, val loss 2.1520
step 3000: train loss 2.0703, val loss 2.1443
step 3500: train loss 2.0619, val loss 2.1201
step 4000: train loss 2.0246, val loss 2.1074
step 4500: train loss 2.0045, val loss 2.1031

And they bridce?

SORORD Edly madisel bube a enamegraves meadied
he ar bady pusque, to bardetlessay, away, my fornot, of oroughtowns, to ficke but milled,
What grive, send, will is therevers, and the now on you meself in you littiser courmby pruperaissell, yet love.
In am paselives hoptell, demeaS?

Ko WICI hour Ceivers the most and To guixends poon of his but that non this fort; igle! mufer, is ale of wachferried my of.
What suk!
Kereby as ardaple,
And he mut in cour as and your to-chave for hi
```

layernorm added and scaling up.

```
step 0: train loss 4.3137, val loss 4.3097
step 500: train loss 2.1006, val loss 2.1587
step 1000: train loss 1.6670, val loss 1.8254
step 1500: train loss 1.4944, val loss 1.6914
step 2000: train loss 1.4017, val loss 1.6121
step 2500: train loss 1.3468, val loss 1.5826
step 3000: train loss 1.3032, val loss 1.5444
step 3500: train loss 1.2681, val loss 1.5278
step 4000: train loss 1.2388, val loss 1.5147
step 4500: train loss 1.2131, val loss 1.5052

God--
But dew hath, and got thy head.
Here come better thy soul bind's kneel.

DUKE OF YORK:
Speak it mock the skindy, you'ld have from thee
That rather, theresome cousing with Juliet,
If rest up in company. Could you should you'll prove
As phemose to chim unevery bart,
To what I be consul. Ong or these that envying
Hath mine collaus.' What say's hare, but this heer!

First You must penitence:
The deep sun suggeer livalry spent,
The more poor, though it layst ripety, and
Ciceive your foe that ig
```