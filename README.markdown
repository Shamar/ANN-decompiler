# "AI" demystified: a decompiler

## Abstract

To prove that any "artificial neural network" is just a statistically
programmed (virtual) machine whose ~~model~~ software is a derivative
work of the source dataset used during its "training", we provide
a small [suite of tools][python-code] to assemble and program such
machines and  a decompiler that reconstruct the source dataset
from the cryptic matrices that constitute the software executed by them.  
Finally we test the suite on the classic [MNIST dataset][mnist]
and compare the decompiled dataset with the original one.

## Keywords

Artificial Intelligence, Machine Learning, Artificial Neural Networks,
Microsoft, GitHub Copilot, Python, Statistical Programming,
Vector Mapping Machine

## Introduction

Despite the hazy anthropomorphic language and the unprofessional
narrative, in the last few years the field of Informatics that goes
under the names of "Artificial Intelligence" and "Machine Learning"
established a long record of self-appointed successes.
Behind the hype and the sci-fi depictions of marvelous futures
where [virtual humans will explore the Universe on behalf of a
mostly extincted humanity][asshole-philosophers] spread to 
[slow down any effective solution][survival-of-the-riches] 
that could save the lifes of billions of real people, there
are few misunderstood (and often misused) techniques that
actually bring some innovation to the table.

The most iconic of such innovations is improperly named "artificial
neural network", after a [wrong model of the human brain][not-neural]
that inspired their creation back in the sixties. On top of this class
of tools, composed and programmed in a multitude of ways, several
useful services are being commoditized by large corporations with
access to huge dataset, money and computational power.

Among them, Microsoft has recently released [GitHub Copilot][copyalot],
a [Service as a Software Substitute][rms-on-saass] that suggests to hasty
programmers code snippets extracted and automatically adapted from the
vast codebase uploaded on GitHub. From the very beginning Microsoft
recognised that [Copilot can recite verbatim code from the GitHub
repositories in 0.1% of the cases][copyalot-parrot], but it didn't take
long before [a developer][mitsuhiko] have been
[suggested][copilot-split-out-quake] a particularly [valuable piece of
free software][quake-copied-source] with a permissive license instead
of the strong copyleft decided by the authors.

After the [obvious backslash][eevee-on-twitter] from the Free Software
community, several lobbists, [lawyers][technollama] and even [some
politicians][julia-reda-on-copilot] run to justify the poor company,
basically arguing that using copyrighted work for "machine learning" is
actually fair, so developers should not annoy the corporation.

As always, the problem is that people who do not really understand
how "machine learning" works, will trust and [spread the
words][misinformed-experts-broadcast] of such misinformed "experts"
and this will inevitably hurt the exploited "humans in the loop".

In fact, Microsoft did **not** use the huge amount of high quality
source code of its own products to train Copilot, but the free software
and the open source software uploaded on GitHub. This despite the fact
that only a tiny minority of projects on GitHub go through the long
reviews and the quality assurances that a company like Microsoft can
afford to set up, and since [Heartbleed][heartbleed] we all know how
the "[Linus's law][linus-law]" is just a bedtime story for naive kids.

## There is no "learning" in "artificial neural network"

Most of the arguments in support of Microsoft's new service comes from
people who really do not understand the inner working of a "artificial
neural network". An ANN is just a virtual machine designed to be
programmed statistically, by examples.

Such kind of virtual machines are composed of tiny devices that are
improperly named "artificial neurons" or "perceptrons", but they are
simply vector [reducers][folding] that, given a non-linear
differentiable function, fold a vector from a certain N-dimensional
space into a 1-dimensional one.

A vector, as you might know, is just a list of numbers that can be
interpreted by a human as a descriptions of something. For example
I might be described by (40, 20, 16), respectively my age, the
number of years I worked as a programmer, and the number of
programming languages I know.

The inner working of a vector reducer is trivial: you just compute
the [dot product][dot-product] between the input vector and a parametric one
and then apply the differentiable function to get the magnitude of the output
vector, but the interesting bit is how you can statistically program the
parametric vector by distributing the observed errors on a known dataset
so that, if you are lucky enough, after some iterations over the dataset 
the vector reducer might approximate the desired output for a given vector.

Such vector reducers can be easily composed in a variety of topologies
and programmed in a variety of ways, so that a whole "artificial neural
network" can be statistically programmed ("trained", in AI/ML parlance)
to approximate one of the possible translations from a vector space to a
different one.

Yet, there is no "learning" ongoing: just the iterative tuning of
parametric vectors to approximate a certain output. Since the various
entries of the parametric vectors are usually called "weights", people
who try to avoid the anthropomorphic jargon of AI usually call the
process "calibration", but as we will see, it's important to use
"statistical programming" instead to clarify to untrained people the
components, the processes, the roles and the responsibilities that
characterize such technology.

Indeed the only "artificial intelligence" in a vector mapping machine
is the one that people are trained to imagine through powerful
marketing, confusing propaganda and a poor technical language.

## Vector Mapping Machines: engineering and programming

Vector mapping machines (improperly known as "artificial neural
network") have been usually implemented as software, so they are
basically specialized (and opaque) **virtual** machines.

In the attempt to preserve the "machine intelligence" pipedream, the
design of such machines has been so much conflated with their
programming, that we needed to invent hyperbolic definitions such as
"hyper-parameters" to let "data scientist" distinquish and comunicate
about the design of a specific machine, its programming (imprerly named
"training") and its inner working at runtime.

However, if we get rid of the sci-fi narrative and talk about opaque
virtual machine specialized in approximated vector reductions, we can
build on the usual language of Informatics to describe and understand
these tools and their impact.

Infact, designing a vector mapping machine is an engineering act
composed of several steps such as deciding the number and kinds of
filters, their connections, the differentiable function to use, the
initialization process of the parametric vectors of each vector reducer
and so on.

The output of such engineering process is a machine that can be built
on hardware or emulated through software, but it's just as useful as a
computer without any software on it.

It's important to note tha when such virtual machine is emulated
throught software, the programming of such emulator obviously grant to
the authors a copyright on it, just like with the programming of any
other virtual machine. But like any other virtual machine, the
copyright holder of the software emulating the machine cannot claim
rights on the software run **by** such machine.

In fact, despite conflated by primitive jergon and practices, the
engineering of vector mapping machines is a totally distinct and
different process from their programming.

The vector mapping machine is a different intellectual artifact from
the program it runs and constitute a creative work on its own. Indeed
the same machine can be programmed to produce different vector mapping
task by providing different source dataset (with the exact same
dimensionality on the input and output vectors).

In other words, an "artificial neural network" **does not learn**
anything and **it's not a network of neurons** or anything like that.
It does **not** understand anything about the data and the output
vectors has **no inherent meaning**: its semantic is always attributed
by humans according to their insights about the statistical program
they uploaded into the vector mapping machine.

As we will see, to program of a vector mapping machine (aka ANN), you
run it to compute an output and then automatically adjusts the
parametric vectors of its reducers backwards, from the last filter to
the first one, so that after each iteration, its output is a bit more
similar to the intended one. It's an iterative process based on
examples whose recurrent relations are "absorbed" by the vector
reducers' parametric vectors and then reproduced at runtime.

That's why it's **statistical programming**: you start with a source
dataset (improperly named "training set" in the AI/ML parlance) and,
after a compilation process that is specifically designed for that
specific virtual machine, you get a binary that such machine can run.

While such binary is not constituted by a sequence of instructions,
it's still just an algorithmic transformation of the original source
that, despite being expressed as cryptic matrices, still contains all
the relevant features of the sources.

The software run by a vector mapping machine is **not a "model"**,
since it does not give any insight about the relations it absorbed
during its statistical programming. Instead it is just a software like
any other that describes a rigorous (if unknown) process that a
specific machine has to follow to automatically and deterministically
compute (or rather **approximate**) a desired output from a given input.

This means for example, that Copilot's "model" is an **algorithmic**
derivative work of all the public repositories uploaded to GitHub and
as such, Microsoft should comply with all network-copyleft that
requires such derivative work to be distributed to the users.

## A Vector Mapping Machine in Python

To show how the "model" of an "artificial neural network" contains
substantial and valuable parts of the sources used to program it, we
assemble and program a simple emulator of a VMM engineered to compute a
classic classification task: the handwritten digits "recognition" from
the MNIST database.

The code is overly simple, avoiding any external dependency except for
the Python standard libary and avoiding fancy mathematical
optimizations that would make the process more difficult to follow, but
it shows how to compile the source dataset into a working binary, how
to check its accuracy and how to decompile it.

While this is a pretty simple "deep learning" proof of concept, it's
always possible (if expensive) to build a compiler/decompiler suite for
any "machine learning" system, including OpenAI and the specific vector
mapping machines operating GitHub's Copilot.

### Assembling a Vector Mapping Machine

The vmm.py module describes the building blocks of a vector mapping machine.

We define `VectorMapper` any device that turns a vector
from a certain vector space (of `inputSize` dimension) into another
vector from another space (of `outputSize` dimension).

An example is the `VectorReducer`, a simple device that constitutes
the basic building block of vector mapping machines: it's the mythical
"perceptron" also known as "artificial neuron" in AI/ML parlance.  
For the sake of simplicity, it embeds the classic [sigmoid][sigmoid]
"transfer" function.

The `ParallelVectorMapper` is a vector mapper that compose several
`VectorReducer` and run them on the same input vector (potentially, in
parallel), returning a multi-dimensional vector composed of their
output.

The `VectorMappingMachine` is just a pipeline of `ParallelVectorMapper`
that can map a vector from the input space in a vector from the output
space by feeding to each filter (aka "layer" in AI/ML parlance) in the
pipeline the output vector returned by the preceding one.

Once we have designed our VMM of choice for the task, we can assemble
it through assemble.py:

```
$ ./assemble.py classifier-784-32-20-10.vm 784 32 20 10
```

will create a "deep learning" vector mapping machine named
`classifier-784-32-20-10.vm` in your current directory.

Once the virtual machine is ready, we need a source of samples to
program it.


### Programming a VMM

To **program** a Vector Mapping Machine **you need a source** dataset.

Google, for example, uses the personal data extracted from unaware
people to program its marketing VRMs just like Microsoft used the
creative work of tons of unaware developers to create GitHub Copilot.

In our case we will simply use the MNIST dataset.

Once you have a source dataset, you often need to turn it into a set of
vectors that you can use to actually program the machine: that's where
several processes such as dimensionality reduction, word embeddings and
so on take place.

In the previous example we created a "classifier" that can map a vector
space with 784 dimensions to a vector space with 10 dimensions: it
could be programmed with any source dataset that match such definition
(784 numeric input features and 10 output categories, with one-hot
encoding for simplicity) but we are going to use the classic MNIST
dataset consisting of 60000 grey-scaled 28x28 images of handwritten
digits.

Since the MNIST database is distributed in a custom binary form and we
want to see how to get back readable sources from the "model" of an
"artificial neural network", the `mnist/` folder contains a
mnist2csv.py script that converts the included dataset into a human
readable CSV: the first 784 integers of each row correspond to the
pixels of a 28x28 raster image while the digit in the last column
corresponds to the intended digit.

The command

```
$ ( cd mnist && ./mnist2csv.py ) 
```

will create `source.csv` and `test.csv` in the mnist/ directory.

A very simple plot.py script (depending on `matplotlib`) is provided
to let you inspect the digit at each sample if you feel so inclined.

### Compiling the sources

Now that we have a source dataset that we can legally use to program
our "digits classifier", we just need to compile it into a binary.

The `compile.py` script produce a binary that can be run by a target
Vector Mapping Machine from a source dataset in CSV format.
The executable format is described by the classes in `ef.py`.

The `loadSamples` procedure loads the rows of the CSV file into a list
of `Sample`, normalizing each column and collecting useful `SourceStats`
such as the maximum and minimum value of each input column and the
ordered list of output categories.
Such statistics will be used at runtime by the VMM to convert the
input CSV rows provided by the users to input vectors composed of
`float` in the 0 - 1 range.

The `compile` procedure is where the parametric vectors of the 
every `VectorReducer` that constitute the machine are iteratively
changed to better approximate the desired output.

Technically speaking, it executes a classic [stochastic gradient
descent][SGD] with [back propagation of errors][backprop].
Simply put, you iterate a few times (aka "epochs") over your
source dataset, try to see what output your VMM produce for each sample,
compute the difference with the expected output and then go backwards
filter by filter, modifying a little the parametric vector of each
reducer so that it would produce a slightly better output given
such input.

There is no magic here, no one is "learning" anything: we are just
iteratively changing the numbers that each vector reducer multiply
for the elements of the input vector so that it will produce an output
more similar to the one we want to obtain.

As we will see later, it's worth noticing that the `compile` procedure
does **not** throw away all informations about how each sample impacts
the final executable, but record a few logs during the last "epoch".


So to compile the source dataset you just need to run

```
$ ./compile.py classifier-784-32-20-10.vm mnist/source.csv mnist/digit-reader.bin
```

After a while (it may take a few of hours on slow machines), you'll get
your program in `mnist/digit-reader.bin`.

Then you can check the program accuracy by running

```
$ ./test.py classifier-784-32-20-10.vm mnist/digit-reader.bin mnist/test.csv
Correct "predictions": 9530/10000 (95.3%)
```

Did our "deep learning" "neural network" "learnt" how to "read"
handwritten digits?

**No.**

We just statistically programmed a Vector Mapping Machine to fold a
vector space with 784 dimensions into a vector space with 10 dimensions.
We did so by leveraging the same kind of regularieties among the source
dataset that the human mind exploits to map a symbol to a digit when such
dataset is plotted into a raster screen, since such regularities are
obviously prominent into a database designed to contain 
"handwritten digits" and they characterize our samples.

Indeed such characteristic regularities are statistically relevant
into the source dataset so they get "absorbed" into the weights of
the reducers that compose the machine during the compilation.

These are fundamental aspects to understand and remember when
thinking about these kind of technologies.

First, the VMM doesn't learn anything: it always reproduces a vector
reduction that is statistically relevant into the source dataset,
just because we statistically programmed it to do so.
In this case, such reduction is possible because **humans designed a
regular set of 10 signs to comunicate digits** and the MNIST source
dataset constains 60 thousands of such signs **drawn by humans for humans**.
To comunicate with other humans, such humans had to reproduce signs
similar to the ones that they would understand and this produce
the regularities that can be "absorbed" and reproduced by a
Vector Mapping Machine.  
In other words, there is no intelligence at work here except
for the human one.

Second, the regularities "absorbed" by the weights of the vector
reducers are **those that characterize the creative work done by
humans**: the creative work that produced the [symbols we use as
numerical digits][hindu-arabic-digits] **and** the creative work
of the people who actually draw those digits  **and**  the work
of people who collected them into the source dataset.
We can legally use these works for different reasons, but at the
end of the day, all of such reasons are based on the consent of
the people who contributed to them.

This is relevant because, whenever the source dataset used to program
a VMM is not composed of data emitted by observed objects or subject
(such as personal data or data about a physical phenomenon), the
executable produced during the compilation is **always a derivative
work** of the source dataset, just like a binary is a derivative
work of the source code used to compile it.

Moreover, in this second case, the program executed by **a VMM/ANN
incorporates exactly those characteristics of the source dataset that
constitutes the creative value of the work**.


## Decompiling (the software run by) an "artificial neural network"

To prove that an "artificial neural network" contains the source dataset
used to "train" it in a cryptic (but not encrypted) form, we will decompile
the executable produced and get back the source dataset with a simple command:

```
$ ./decompile.py mnist/digit-reader.bin mnist/digit-reader.sources.csv
```

This will create a new CSV file named `digit-reader.sources.csv` that
can be compared to the MNIST dataset in `mnist/sources.csv`.

In our example, 59993 decompiled samples perfectly match the one from
the original MNIST dataset (~99.99%), and **only seven images does not
match perfectly the original ones**, but the differences are
imperceptible to the eye. Infact, in those images, the slight
variations on the gray scale of some pixels are due to rounding errors
during floating point operations. Such errors grow with the raise of
the "prediction accuracy" of the executable, but they are just
implementations issues that could be fixed by using arbitrary precision
floating-point arithmetic libraries.


![The seven mismatches between the source dataset and the decompiled one.](http://www.tesio.it/2021/09/01/differences.png "The seven mismatches between the source dataset and the decompiled one.")


So we can see that the program executed by the "artificial neural
network" actually contains the whole "training" dataset despite
arranged in a pretty cryptic set of numerical matrices.

Most of the decompilation process occurs in the `decompile` procedure
that traverse the logs recorded during the last iteration over the
source dataset (the last "epoch" in AI/ML parlance) and revert the
compilation process.

For each log line, the VMM is traversed backwards, from the last filter
to the first one.

For each filter we use the recorded output, the error and the weights'
variations of a sentinel node to compute the input vector received from
the preceeding filter. Then we compute the weight variations for every
other reducer of the filter and apply them. 

Now, if you are too fond in the AI narrative, if you worship Futurology
or if you are a [transumanist][bullshit-of-the-rich-but-for-the-masses]
scared about imaginary "existential risks", you should be seriously
outraged here: the "neural network" is "unlearning"!  
We are literally "brain washing" the machine!  
It could happen to your imaginary simulated-heirs too!  
And there's nothing you can do about it! :-PP

More seriously, when all the reducers of the filters have been restored
to the previous parameters, we compute the errors for the previous
filter and then repeat the process with it.

Finally, when all the filters have been processed, the input of the
first filter is rescaled and saved in the output csv and we move to the
next log.

## Where is the trick?

Usually, during the compilation process (aka "training"), each sample
from the source dataset is turned into a miriad of numeric vectors that
are "absorbed" by the parametric vectors of the reducers in the
machine... and then discarded.

By recording very few of such vectors (one for each filter, to be
precise) **and** reverting the compilation process, we can compute all
the other vectors and the original input.

In particular, for each sample compiled during the last iteration
over the dataset, the executable will retain:

- the output vector produced by the VMM
- the error vector (the difference between the computed output and the
  one provided by the sample sources)
- the variation of weights applied to **one single** `VectorReducer` for
  each `ParallelVectorMapper` of the program.

It does not matter how complex or large is a "neural network",
the changes to the parametric vector of a single "neuron" of each "layer"
is enough.

Note that such vectors are not enough, by themselves, to get back the
original dataset: the final weights used by the reducers that compose
the machine are also needed during the decompilation process because
they hold fundamental projections of the dataset.

In other words, any "model" of an "AI" is a substantial portion of a
larger derivative work based on the dataset used during the "training".

Discarding the rest of the work (the logs that `compiler.py` preserved)
is just a way to hide the proofs of such derivation.

That's why nobody should be surprised when GitHub Copilot ~~distributes~~
suggests to include copylefted code into proprietary software: it has
literally been "trained" to violate hackers' copyright from the very
beginning!

In fact, it still contains substantial portions of the copylefted
code uploaded on GitHub, but it does not comply with the licenses.

# Conclusions

"GitHub Copilot" should be probably rebranded as "Microsoft Copy(a)lot". ;-)

Its software, like any other "model" produced though those techniques
that currently go under the umbrella of "artificial intelligence", is a
derivative work of the source dataset, just like any binary produced by
a compiler is a derivative work of the sources.

Thus Microsoft should either get a license from the authors of the
sources they compiled into such software, or comply with **all** their
licenses.

On the other hand, if Microsoft is right and programming a Vector
Mapping Machine is enough to remove the copyrights from a work, we
could use the technique shown here to compile texts, songs, movies and
even proprietary software into a program for a standard VMM and then
decompile it when required, to play or execute it: the "artificial
intelligence" would remove any legal risk for the people distributing
the compiled binary.

In fact, if what Microsoft did was actually "fair use", every work under
copyright can be used in the same way. In other words, if Microsoft is
right, they made obsolete most of the [oscurantist "intellectual
property" laws][ip] that plague our cybernetic society: a huge step
forward that we should all welcome with gratitude.

# Further research

The technique shown here is pretty simple but it can be adapted to any
other "machine learning" technique. You just need to **not throw away**
relevant data during the compilation of the source dataset.

Furthermore, such technique could be extended to verify, under
appropriate conditions, if and how a certain set of samples has been
used during the compilation of a given "artificial intelligence" in a
forensic analysis.

Finally, if the required data has been dutyfully preserved during the realization
of a "machine learning" system, such technique could be adapted to 
**remove problematic samples** from the program executed from the system.

# Licensing

[This essay][canonical-uri] and the all [the Python code][python-code] distribuited with it is licensed
to you under the terms of the [Hacking License][hacking-license]
(see [HACK.txt](HACK.txt)).

The MNIST dataset has been copied in [the repository][copilot-trojan]
as requested by its [maintainers][mnist].

[canonical-uri]: http://www.tesio.it/2021/09/01/a_decompiler_for_artificial_neural_networks.html

[python-code]: http://www.tesio.it/2021/09/01/ann-decompiler.tar.gz

[copilot-trojan]: https://github.com/Shamar/ANN-decompiler

[copyalot]: https://copilot.github.com/

[copyalot-parrot]: https://docs.github.com/en/github/copilot/research-recitation

[mnist]: http://yann.lecun.com/exdb/mnist/

[asshole-philosophers]: https://www.currentaffairs.org/2021/07/the-dangerous-ideas-of-longtermism-and-existential-risk

[rms-on-saass]: https://www.gnu.org/philosophy/who-does-that-server-really-serve.html.en

[copilot-split-out-quake]: https://video.twimg.com/tweet_video/E5R5lsfXoAQDRkE.mp4

[mitsuhiko]: https://lucumr.pocoo.org/

[quake-copied-source]: https://github.com/id-Software/Quake-III-Arena/blob/dbe4ddb10315479fc00086f08e25d968b4b43c49/code/game/q_math.c#L552-L572

[eevee-on-twitter]: https://twitter.com/eevee/status/1410037309848752128

[julia-reda-on-copilot]: https://juliareda.eu/2021/07/github-copilot-is-not-infringing-your-copyright/

[heartbleed]: https://heartbleed.com/

[linus-law]: https://en.wikipedia.org/wiki/Linus%27s_law

[folding]: https://en.wikipedia.org/wiki/Fold_(higher-order_function)

[copilot-faq-pr-change]: https://twitter.com/alexjc/status/1410524416660889607

[alphago-zero]: http://discovery.ucl.ac.uk/10045895/1/agz_unformatted_nature.pdf

[hindu-arabic-digits]: https://en.wikipedia.org/wiki/History_of_the_Hindu%E2%80%93Arabic_numeral_system

[misinformed-experts-broadcast]: https://thenextweb.com/news/github-copilot-ai-copyright-analysis

[technollama]: https://www.technollama.co.uk/is-githubs-copilot-potentially-infringing-copyright

[ip]: https://locusmag.com/2020/09/cory-doctorow-ip/

[mpmath]: https://mpmath.org/

[not-neural]: https://sci-hub.st/10.1038/337129a0

[survival-of-the-riches]: https://onezero.medium.com/survival-of-the-richest-9ef6cddd0cc1

[SGD]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent

[backprop]: https://en.wikipedia.org/wiki/Backpropagation

[sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function

[bullshit-of-the-rich-but-for-the-masses]: https://en.wikipedia.org/wiki/Transhumanism

[dot-product]: https://en.wikipedia.org/wiki/Dot_product

[hacking-license]: http://www.tesio.it/documents/HACK.txt