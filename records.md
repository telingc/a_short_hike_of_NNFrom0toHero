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