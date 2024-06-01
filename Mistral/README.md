# Mistral
This repository includes a pytorch implementation of Mistral models (Base + Mixture Of Experts). The dataset used in the 
training of the model is based on dialogues from a play. The model is a miniature version of the actual Mistral due to 
infrastructure constraints.

## Mistral Base

* Epochs = 30
* Batch Size = 32
* Max Seq Len = 512
* Window Size = 128
* Dim = 128
* Decoder Blocks = 2
* Total Heads = 8
* Total KV Heads = 2
* Vocab Size = 32001

### Training Dataset
The training dataset is same as the one present inside the Llama2 folder. Check the file named **input.txt**.

### Results

* Text Generation
```
Test Sample 1:
compression, and witherous,
And then appointed it on the world,
Whereby, as I was a true-bed;
And I'll be a feast of mine.

QUEEN MARGARET:
I am dead; and I'll tell you to my breast,
And I am a traitor, I'll to be thee.

QUEEN MARGARET:
I'll stay thee in a golden fee-fath
To seek thee to thy friend of death.

QUEEN ELIZABETH:
And I'll prove a widowbray,
And that thou hast sworn'd me from thee,
And thou wilt perform thee to thy head;
And thou, that thou art not yet drunk:
And thou, for thy father's death,
And thou not thy father's death, and thou wert not thy
Thy father's death of death dost thou wert not
To seek thee with thee,
And thou wilt object thee, and thou wert not
To seek thee in thy head of death.

DUKE OF AUMERLE:
I am a traitor, and my son,
And, and let me see thee,
And that thou hast done to thy words:
I have a thousand years to-night and see,
And that thou hast thy hell'd me, thou wert not thy woe?

KING RICHARD II:
I am a man.

QUEEN MARGARET:
So, I'll not be to thee to my crown:
The crown I have to my son to-day,
And I'll make a manors to my fortune,
And I'll to be thee to my hands.

QUEEN MARGARET:
Thy brother, my lords, and I am dead;
And I, for thy kinsmanly for me,
And thou thee to thy glory and myself.

QUE:
God keep thee, my lord;
And I will not stay thee in thee,
And that thou hast thee to thy weapon,
And thou to thy father with thee?

GLO
```

* Prompt Completion
```
Prompt:

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they


Result:

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
are as soon as the other things and the people
are.
 
Second Citizen:
I have done
The people.
 
Second Citizen:
I have:
The people's off.
 
Second Citizen:
I have:
I have heard of your voices, and have
ed them to the people.
 
First Citizen:
I have heard the people?

Second Citizen:
I have:
The people, and the people, and the people,
loves to the people.

Second Citizen:
I' the people, I'll warrant him to the people,
and the people, and the people, and
the people, and the people, and
The gods, and the people, and
themselves, to the people, and to make the people,
And to be consul, and to make the people,
The other way from the people, and they say,
The to end motion of the people, and they say,
The extreme hath made a man of the people,
Which, to the people, and with the people,
And, by the people, and
Have lost a soldier, and in the ends,
To see the other's and the city,
And in the city of the city,
And, by the other's hear me,
And, by the city, the third, and,
To take upon the sea, and in the ends,
To see the other's front.

MENENIUS:
I'll have the gods,
And bring him their voices.

MENENIUS:
I think there's but the general which,
I'll have him to thee in the Capitol,
And the gods, the whole army of the Capitol,
And thee, the whole army of the Capitol,
And in thee to thee,
And in thee to thee,
And, to thee, to thee, to thee
And in thee, to thee, to thee
For usurge and by thee,
And then, to thee, to thee,
And
```

## Mistral With Mixture Of Experts

* Epochs = 50
* Batch Size = 32
* Max Seq Len = 512
* Window Size = 128
* Dim = 128
* Decoder Blocks = 2
* Total Heads = 8
* Total KV Heads = 2
* Vocab Size = 32001
* Num Of Experts = 2
* Top K Experts = 1

### Training Dataset
The training dataset is same as the one present inside the Llama2 folder. Check the file named **input.txt**.

### Results

* Text Generation
```
Test Sample 1:
것.
 
HERMIONE:
Well, I say, I am not?

DUKE VINCENTIO:
What is the matter?

Provost:
I shall go play
A ofman, sir, sir.

DUKE VINCENTIO:

ProvostILLIUS:
I was not, sir.

DORCENTIO:
If you have a man so?

AUTOLYCUS:
A hundred, I pray, of you, sir.

Clown:
Behold youth, sir.

ESCALUS:
How now!

ESCALUS:

DORCAS!

AUTOLYCUS:

DORCAS:
What is the matter?

AUTOLYCUS:
A hundred-me, a most-ma!'

MOPSA:
Amen.

AUTOLYCUS:

AUTOLYCUS:

AUTOLYCUS:

Clown:
I' the absentold.

MOPSA:
How?

AUT:
I' the lark, sir.

MOPSA:
How?

AUTOLYCUS:
A hundred, sir.

DORCAS:
How?

AUTOLYCUS:
A hundred mad!

AUTOLYCUS:
This, a standings, and a most man's
ones, a seems, for that receest advocateing both.

MOPSA:
A pair of fourscore out: and he was on his
pawn well.

DORCAS:
How?

AUTOLYCUS:
A very true; and a man that ballad in prison.

AUTOLYCUS:
I know you well.

AUTOLIO:
I' the ask, sir.

AUTOLYCUS:
This is the tune to have no
to prison?

AUTOLYCUS:
This nor face is coming.

Clown:
His garments, sir.
```

* Prompt Completion
```
Prompt:

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they


Result:

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
false of any other issue of victory.
 
Second Gentleman:
No, but they will, as many, as you know, as you
have not found their heads.

CORIOLANUS:
No, my good lord.

MENENIUS:
No, what of that?

SICINIUS:
Ay, what of that?

CORIOLANUS:
Ay, what are you then?

MENENIUS:
I do not go on the Volscian: I say.

MENENIUS:
I shall, as we were better about me
To answer to the people: he had I'll look
your finger with Aufidels, and the chest note
Into your own: he shall chance after him.

MENENIUS:
'Twas authority.

MENENIUS:
It will not.

CORIOLANUS:
You have been too noble and
As I have stood after's house.

MENENIUS:
Sir, what's the news?

MENENIUS:
I'll not mark me.

CORIOLANUS:
Well?

MENENIUS:
I'll hear you.

MENENIUS:
Sir, what's the matter?

MENENIUS:
Sir, what's.

MENENIUS:
Behold: he'sell a Romano the common anon of us
occasion.

MENENIUS:
Sir, what's in me?

MENENIUS:
Marcius I know; and we'll show'd
To help him in hope.

MENENIUS:
Say, what of that we do lie that;
The which of us was coming'd?

MENENIUS:
I'll not medd them.

MENENIUS:
Sir, I
```
