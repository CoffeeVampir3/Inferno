comptime prompt_one = """Constantinos Daskalakis (Greek: Κωνσταντίνος Δασκαλάκης; born 29 April 1981) is a Greek theoretical computer scientist.[2] He is a professor at MIT's Electrical Engineering and Computer Science department and a member of the MIT Computer Science and Artificial Intelligence Laboratory.[3][4][5] He was awarded the Rolf Nevanlinna Prize and the Grace Murray Hopper Award in 2018.
Education and career

Daskalakis was born in Athens on 29 April 1981.[6] His grandparents originated from Crete, where he summered as a child. His parents were high school teachers of mathematics and literature.[7][8] He has a younger brother, Nikolaos Daskalakis, who is a neuroscientist and Boston University professor.[9][10] When Daskalakis was in third grade, his father bought an Amstrad CPC, which Daskalakis stayed up all night with, attempting to learn how it worked.[11]

He attended Varvakeio High School and received a Diploma in Electrical and Computer Engineering from the National Technical University of Athens in 2004, completing an undergraduate thesis supervised by Stathis Zachos. As an undergraduate, Daskalakis attained perfect scores in all but one of his classes, something which had not previously been achieved in the university's history.[11] He received a PhD in computer science from the University of California, Berkeley advised by Christos Papadimitriou.[12][1]

From 2008 to 2009, Daskalakis was a postdoctoral researcher at Microsoft Research mentored by Jennifer Chayes. He joined MIT in 2009 and was given tenure in 2015.[13]

He is a co-founder and chief scientist of Archimedes AI research center.[14]
Research

Daskalakis works on the theory of computation and its interface with game theory, economics, probability theory, statistics and machine learning.[2] He is known for work on the computational complexity of Nash equilibria, the complexity of multi-item auctions, and the behavior of the expectation–maximization algorithm. He has worked on efficient methods for statistical hypothesis testing and learning in high dimensions, as well as concentration properties of high-dimensional distributions.
Awards and honors

Constantinos Daskalakis was awarded the 2008 ACM Doctoral Dissertation Award for "advancing our understanding of behavior in complex networks of interacting individuals."[15] He later co-authored the paper The Complexity of Computing a Nash Equilibrium[16] based on the same work with Christos Papadimitriou and Paul W. Goldberg, for which they were awarded the 2008 Kalai Game Theory and Computer Science Prize.[17]

In 2018, Daskalakis was awarded the Nevanlinna Prize for "transforming our understanding of the computational complexity of fundamental problems in markets, auctions, equilibria and other economic structures."[18] In the same year, he also received the Simons Foundation Investigator award in theoretical computer science.[19]

He was named to the 2022 class of ACM Fellows "for fundamental contributions to algorithmic game theory, mechanism design, sublinear algorithms, and theoretical machine learning".[20] """

comptime prompt_two = """In computer science, the computational complexity or simply complexity of an algorithm is the amount of resources required to run it.[1] Particular focus is given to computation time (generally measured by the number of needed elementary operations) and memory storage requirements. The complexity of a problem is the complexity of the best algorithms that allow solving the problem.

The study of the complexity of explicitly given algorithms is called analysis of algorithms, while the study of the complexity of problems is called computational complexity theory. Both areas are highly related, as the complexity of an algorithm is always an upper bound on the complexity of the problem solved by this algorithm. Moreover, for designing efficient algorithms, it is often fundamental to compare the complexity of a specific algorithm to the complexity of the problem to be solved. Also, in most cases, the only thing that is known about the complexity of a problem is that it is no higher than the complexity of the most efficient known algorithms. Therefore, there is a large overlap between analysis of algorithms and complexity theory.

As the amount of resources required to run an algorithm generally varies with the size of the input, the complexity is typically expressed as a function n ↦ f(n), where n is the size of the input and f(n) is either the worst-case complexity (the maximum of the amount of resources that are needed over all inputs of size n) or the average-case complexity (the average of the amount of resources over all inputs of size n). Time complexity is generally expressed as the number of required elementary operations on an input of size n, where elementary operations are assumed to take a constant amount of time on a given computer and change only by a constant factor when run on a different computer. Space complexity is generally expressed as the amount of memory required by an algorithm on an input of size n.
Resources
Time

The resource that is most commonly considered is time. When "complexity" is used without qualification, this generally means time complexity.

The usual units of time (seconds, minutes etc.) are not used in complexity theory because they are too dependent on the choice of a specific computer and on the evolution of technology. For instance, a computer today can execute an algorithm significantly faster than a computer from the 1960s; however, this is not an intrinsic feature of the algorithm but rather a consequence of technological advances in computer hardware. Complexity theory seeks to quantify the intrinsic time requirements of algorithms, that is, the basic time constraints an algorithm would place on any computer. This is achieved by counting the number of elementary operations that are executed during the computation. These operations are assumed to take constant time (that is, not affected by the size of the input) on a given machine, and are often called steps.
Bit complexity

Formally, the bit complexity refers to the number of operations on bits that are needed for running an algorithm. With most models of computation, it equals the time complexity up to a constant factor. On computers, the number of operations on machine words that are needed is also proportional to the bit complexity. So, the time complexity and the bit complexity are equivalent for realistic models of computation.
Space

Another important resource is the size of computer memory that is needed for running algorithms.
Circuit
Main article: circuit complexity
Communication
Main article: communication complexity

For the class of distributed algorithms that are commonly executed by multiple, interacting parties, the resource that is of most interest is the communication complexity. It is the necessary amount of communication between the executing parties.
Others

The number of arithmetic operations is another resource that is commonly used. In this case, one talks of arithmetic complexity. If one knows an upper bound on the size of the binary representation of the numbers that occur during a computation, the time complexity is generally the product of the arithmetic complexity by a constant factor.

For many algorithms the size of the integers that are used during a computation is not bounded, and it is not realistic to consider that arithmetic operations take a constant time. Therefore, the time complexity, generally called bit complexity in this context, may be much larger than the arithmetic complexity. For example, the arithmetic complexity of the computation of the determinant of a n×n integer matrix is O ( n 3 ) {\\\\displaystyle O(n^{3})} for the usual algorithms (Gaussian elimination). The bit complexity of the same algorithms is exponential in n, because the size of the coefficients may grow exponentially during the computation. On the other hand, if these algorithms are coupled with multi-modular arithmetic, the bit complexity may be reduced to Õ(n4).

In sorting and searching, the resource that is generally considered is the number of entry comparisons. This is generally a good measure of the time complexity if data are suitably organized.
Complexity as a function of input size
Only time complexity is considered in this section, but everything applies (with slight modifications) to the complexity with respect to other resources

It is impossible to count the number of steps of an algorithm on all possible inputs. As the complexity generally increases with the size of the input, the complexity is typically expressed as a function of the size n (in bits) of the input, and therefore, the complexity is a function of n. However, the complexity of an algorithm may vary dramatically for different inputs of the same size. Therefore, several complexity functions are commonly used.

The worst-case complexity is the maximum of the complexity over all inputs of size n, and the average-case complexity is the average of the complexity over all inputs of size n (this makes sense, as the number of possible inputs of a given size is finite). Generally, when "complexity" is used without being further specified, it is the worst-case time complexity that is considered.
Asymptotic complexity
Main article: Asymptotic computational complexity

It is generally difficult to compute precisely the worst-case and the average-case complexity. In addition, these exact values provide little practical application, as any change of computer or of model of computation would change the complexity somewhat. Moreover, the resource use is not critical for small values of n, and this makes that, for small n, the ease of implementation is generally more interesting than a low complexity.

For these reasons, one generally focuses on the behavior of the complexity for large n, that is on its asymptotic behavior when n tends to the infinity. Therefore, the complexity is generally expressed by using big O notation.

For example, the usual algorithm for integer multiplication has a complexity of O ( n 2 ) ; {\\\\displaystyle O(n^{2});} this means that there is a constant c u {\\\\displaystyle c_{u}} such that the multiplication of two integers of at most n digits may be done in a time less than c u n 2 . {\\\\displaystyle c_{u}n^{2}.} This bound is sharp in the sense that the worst-case complexity and the average-case complexity are Ω ( n 2 ) , {\\\\displaystyle \\\\Omega (n^{2}),} which means that there is also a constant c l {\\\\displaystyle c_{l}} such that these complexities are larger than c l n 2 . {\\\\displaystyle c_{l}n^{2}.} The radix does not appear in these complexities, as changing of radix changes only the constants c u {\\\\displaystyle c_{u}} and c l . {\\\\displaystyle c_{l}.}
Models of computation

The evaluation of the complexity relies on the choice of a model of computation, which consists in defining the basic operations that are done in a unit of time. When the model of computation is not explicitly specified, it is generally implicitly assumed to be a multitape Turing machine, since several more realistic models of computation, such as random-access machines are asymptotically equivalent for most problems. It is only for very specific and difficult problems, such as integer multiplication in time O ( n log ⁡ n ) , {\\\\displaystyle O(n\\\\log n),} that the explicit definition of the model of computation is required for proofs.
Deterministic models

A deterministic model of computation is a model of computation such that the successive states of the machine and the operations to be performed are completely determined by the preceding state. Historically, the first deterministic models were recursive functions, lambda calculus, and Turing machines. The model of random-access machines (also called RAM-machines) is also widely used, as a closer counterpart to real computers.

When the model of computation is not specified, it is generally assumed to be a multitape Turing machine. For most algorithms, the time complexity is the same on multitape Turing machines as on RAM-machines, although some care may be needed in how data is stored in memory to get this equivalence.
Non-deterministic computation

In a non-deterministic model of computation, such as non-deterministic Turing machines, some choices may be made at some steps of the computation. In complexity theory, one considers all possible choices simultaneously, and the non-deterministic time complexity is the time needed, when the best choices are always made. In other words, one considers that the computation is done simultaneously on as many (identical) processors as needed, and the non-deterministic computation time is the time spent by the first processor that finishes the computation. This parallelism is partly amenable to quantum computing via superposed entangled states in running specific quantum algorithms, like e.g. Shor's algorithm for finding the prime factors of an integer. """

comptime prompt_three = """Photosynthesis[note 1] is a system of biological processes by which photopigment-bearing autotrophic organisms, such as most plants, algae and cyanobacteria, convert light energy — typically from sunlight — into the chemical energy necessary to fuel their metabolism. The term photosynthesis usually refers to oxygenic photosynthesis, a process that releases oxygen as a byproduct of water splitting. Photosynthetic organisms store the converted chemical energy within the bonds of intracellular organic compounds (complex compounds containing carbon), typically carbohydrates like sugars (mainly glucose, fructose and sucrose), starches, phytoglycogen and cellulose. When needing to use this stored energy, an organism's cells then metabolize the organic compounds through cellular respiration. Photosynthesis plays a critical role in producing and maintaining the oxygen content of the Earth's atmosphere, and it supplies most of the biological energy necessary for complex life on Earth.[2]

Some organisms also perform anoxygenic photosynthesis, which does not produce oxygen. Some bacteria (e.g. purple bacteria) use bacteriochlorophyll to split hydrogen sulfide as a reductant instead of water, releasing sulfur instead of oxygen, which was a dominant form of photosynthesis in the euxinic Canfield oceans during the Boring Billion.[3][4] Archaea such as Halobacterium also perform a type of non-carbon-fixing anoxygenic photosynthesis, where the simpler photopigment retinal and its microbial rhodopsin derivatives are used to absorb green light and produce a proton (hydron) gradient across the cell membrane, and the subsequent ion movement powers transmembrane proton pumps to directly synthesize adenosine triphosphate (ATP), the "energy currency" of cells. Such archaeal photosynthesis might have been the earliest form of photosynthesis that evolved on Earth, as far back as the Paleoarchean, preceding that of cyanobacteria (see Purple Earth hypothesis).[5]

While the details may differ between species, the process always begins when light energy is absorbed by the reaction centers, proteins that contain photosynthetic pigments or chromophores. In plants, these pigments are chlorophylls (a porphyrin derivative that absorbs the red and blue spectra of light, thus reflecting green) held inside chloroplasts, abundant in leaf cells. In cyanobacteria, they are embedded in the plasma membrane. In these light-dependent reactions, some energy is used to strip electrons from suitable substances, such as water, producing oxygen gas. The hydrogen freed by the splitting of water is used in the creation of two important molecules that participate in energetic processes: reduced nicotinamide adenine dinucleotide phosphate (NADPH) and ATP.

In plants, algae, and cyanobacteria, sugars are synthesized by a subsequent sequence of light-independent reactions called the Calvin cycle. In this process, atmospheric carbon dioxide is incorporated into already existing organic compounds, such as ribulose bisphosphate (RuBP).[6] Using the ATP and NADPH produced by the light-dependent reactions, the resulting compounds are then reduced and removed to form further carbohydrates, such as glucose. In other bacteria, different mechanisms like the reverse Krebs cycle are used to achieve the same end.

The first photosynthetic organisms probably evolved early in the evolutionary history of life using reducing agents such as hydrogen or hydrogen sulfide, rather than water, as sources of electrons.[7] Cyanobacteria appeared later; the excess oxygen they produced contributed directly to the oxygenation of the Earth,[8] which rendered the evolution of complex life possible. The average rate of energy captured by global photosynthesis is approximately 130 terawatts,[9][10][11] which is about eight times the total power consumption of human civilization.[12] Photosynthetic organisms also convert around 100–115 billion tons (91–104 Pg petagrams, or billions of metric tons), of carbon into biomass per year.[13][14] Photosynthesis was discovered in 1779 by Jan Ingenhousz who showed that plants need light, not just soil and water. """

comptime prompt_four = """A chloroplast (/ˈklɔːrəˌplæst, -plɑːst/ KLOR-ə-plast, -⁠plahst)[1][2] is a type of organelle known as a plastid that conducts photosynthesis mostly in plant and algal cells. Chloroplasts have a high concentration of chlorophyll pigments which capture the energy from sunlight and convert it to chemical energy and release oxygen. The chemical energy created is then used to make sugar and other organic molecules from carbon dioxide in a process called the Calvin cycle. Chloroplasts carry out a number of other functions, including fatty acid synthesis, amino acid synthesis, and the immune response in plants. The number of chloroplasts per cell varies from one, in some unicellular algae, up to 100 in plants like Arabidopsis and wheat.

Chloroplasts are highly dynamic—they circulate and are moved around within cells. Their behavior is strongly influenced by environmental factors like light color and intensity. Chloroplasts cannot be made anew by the plant cell and must be inherited by each daughter cell during cell division, which is thought to be inherited from their ancestor—an ancient photosynthetic cyanobacterium that was engulfed by an early eukaryotic cell.[3]

Because of their endosymbiotic origins, chloroplasts, like mitochondria, contain their own DNA separate from the cell nucleus. With one exception (the amoeboid Paulinella chromatophora), all chloroplasts can be traced back to a single endosymbiotic event. Despite this, chloroplasts can be found in extremely diverse organisms that are not directly related to each other—a consequence of many secondary and even tertiary endosymbiotic events.
Discovery and etymology

The first definitive description of a chloroplast (Chlorophyllkörnen, "grain of chlorophyll") was given by Hugo von Mohl in 1837 as discrete bodies within the green plant cell.[4] In 1883, Andreas Franz Wilhelm Schimper named these bodies as "chloroplastids" (Chloroplastida).[5] In 1884, Eduard Strasburger adopted the term "chloroplasts" (Chloroplasten).[6][7][8]

The word chloroplast is derived from the Greek words chloros (χλωρός), which means green, and plastes (πλάστης), which means "the one who forms".[9]
Endosymbiotic origin of chloroplasts
Main article: Plastid evolution
See also: Cyanobacteria and Symbiogenesis

Chloroplasts are one of many types of organelles in photosynthetic eukaryotic cells. They evolved from cyanobacteria through a process called organellogenesis.[10] Cyanobacteria are a diverse phylum of gram-negative bacteria capable of carrying out oxygenic photosynthesis. Like chloroplasts, they have thylakoids.[11] The thylakoid membranes contain photosynthetic pigments, including chlorophyll a.[12][13] This origin of chloroplasts was first suggested by the Russian biologist Konstantin Mereschkowski in 1905[14] after Andreas Franz Wilhelm Schimper observed in 1883 that chloroplasts closely resemble cyanobacteria.[5] Chloroplasts are only found in plants, algae,[15] and some species of the amoeboid Paulinella.[16]

Mitochondria are thought to have come from a similar endosymbiosis event, where an aerobic prokaryote was engulfed.[17]
"""

comptime prompt_five = """Hugo von Mohl FFRS HFRSE (8 April 1805 – 1 April 1872) was a German botanist from Stuttgart. He was the first person to use the word "protoplasm" for the part of a cell inside the cell wall.[1]
Life

He was a son of the Württemberg statesman Benjamin Ferdinand von Mohl [de] (1766–1845), the family being connected on both sides with the higher class of state officials of Württemberg. While a pupil at the gymnasium, he pursued botany and mineralogy in his leisure time, until in 1823 he entered the University of Tübingen. After graduating with distinction in medicine he went to Munich, where he met a distinguished circle of botanists, and found ample material for research.[2]

This seems to have determined his career as a botanist, and he started in 1828 those anatomical investigations which continued until his death. In 1832 he was appointed professor of botany in Tübingen, a post which he never left. Unmarried, his pleasures were in his laboratory and library, and in perfecting optical apparatus and microscopic preparations, for which he showed extraordinary manual skill. He was largely a self-taught botanist from boyhood, and, little influenced in his opinions even by his teachers, preserved always his independence of view on scientific questions. He received many honours during his lifetime, and was elected foreign fellow of the Royal Society in 1868.[2]

The process of cell division as observed under a microscope was first discovered by Hugo von Mohl in 1835 as he worked on green algae Cladophora glomerata.[3]

Mohl's writings cover a period of forty-four years; the most notable of them were republished in 1845 in a volume entitled Vermischte Schriften (For lists of his works see Botanische Zeitung, 1872, p. 576, and Royal Soc. Catalogue, 1870, vol. iv.) They dealt with a variety of subjects, but chiefly with the structure of the higher forms, including both rough anatomy and minute histology. The word protoplasm was his suggestion; the nucleus had already been recognized by R. Brown and others; but Mohl showed in 1844 that the protoplasm is the source of those movements which at that time excited so much attention.[2]

He recognized under the name of primordial utricle the protoplasmic lining of the vacuolated cell, and first described the behaviour of the protoplasm in cell division. These and other observations led to the overthrow of Schleiden's theory of origin of cells by free-cell-formation. His contributions to knowledge of the cell-wall were no less remarkable; he held the view now generally adopted of growth of cell-wall by apposition. He first explained the true nature of pits, and showed the cellular origin of vessels and of fibrous cells; he was, in fact, the true founder of the cell theory. Clearly the author of such researches was the man to collect into one volume the theory of cell-formation, and this he did in his treatise Die vegetabilische Zelle (1851), a short work translated into English (Ray Society, 1852).[2]

Mohl's early investigations on the structure of palms, of cycads, and of tree ferns permanently laid the foundation of all later knowledge of this subject: so also his work on Isoetes (1840). His later anatomical work was chiefly on the stems of dicotyledons and gymnosperms; in his observations on cork and bark he first explained the formation and origin of different types of bark, and corrected errors relating to lenticels. Following on his early demonstration of the origin of stomata (1838), he wrote a classical paper on their opening and closing (1850).[2]

In 1843 he started the weekly Botanische Zeitung in conjunction with Schlechtendal, which he edited jointly until his death. He was never a great writer of comprehensive works; no textbook exists in his name, and it would indeed appear from his withdrawal from co-operation in Hofmeister's Handbuch that he had a distaste for such efforts.[2] In 1850, he was elected a foreign member of the Royal Swedish Academy of Sciences.[citation needed] In his latter years his productive activity fell off, doubtless through failing health, and he died suddenly at Tübingen on 1 April 1872.[2] """

comptime prompt_six = """A plastid is a membrane-bound organelle found in the cells of plants, algae, and some other eukaryotic organisms. Plastids are considered to be intracellular endosymbiotic cyanobacteria.[1]

Examples of plastids include chloroplasts (used for photosynthesis); chromoplasts (used for synthesis and storage of pigments); leucoplasts (non-pigmented plastids, some of which can differentiate); and apicoplasts (non-photosynthetic plastids of apicomplexa derived from secondary endosymbiosis).

A permanent primary endosymbiosis event occurred about 1.5 billion years ago in the Archaeplastida clade—land plants, red algae, green algae and glaucophytes—probably with a cyanobiont, a symbiotic cyanobacteria related to the genus Gloeomargarita.[2][3]

Another primary endosymbiosis event occurred later, between 140 and 90 million years ago, in the photosynthetic plastids Paulinella amoeboids of the cyanobacteria genera Prochlorococcus and Synechococcus, or the "PS-clade".[4][5]

Secondary and tertiary endosymbiosis events have also occurred in a wide variety of organisms; and some organisms developed the capacity to sequester ingested plastids—a process known as kleptoplasty.

Andreas Schimper[6][a] was the first to name, describe, and provide a clear definition of plastids, which possess a double-stranded DNA molecule that long has been thought of as circular in shape, like that of the circular chromosome of prokaryotic cells—but now, perhaps not; (see "..a linear shape"). Plastids are sites for manufacturing and storing pigments and other important chemical compounds used by the cells of autotrophic eukaryotes. Some contain biological pigments such as used in photosynthesis or which determine a cell's color. Plastids in organisms that have lost their photosynthetic properties are highly useful for manufacturing molecules like the isoprenoids.[8]
In land plants
Plastid types
Leucoplasts in plant cells.
Chloroplasts, proplastids, and differentiation

In land plants, the plastids that contain chlorophyll can perform photosynthesis, thereby creating internal chemical energy from external sunlight energy while capturing carbon from Earth's atmosphere and furnishing the atmosphere with life-giving oxygen. These are the chlorophyll-plastids—and they are named chloroplasts; (see top graphic).

Other plastids can synthesize fatty acids and terpenes, which may be used to produce energy or as raw material to synthesize other molecules. For example, plastid epidermal cells manufacture the components of the tissue system known as plant cuticle, including its epicuticular wax, from palmitic acid—which itself is synthesized in the chloroplasts of the mesophyll tissue. Plastids function to store different components including starches, fats, and proteins.[9]

All plastids are derived from proplastids (also named proplasts[10]), which are present in the meristematic regions of the plant. Proplastids and young chloroplasts typically divide by binary fission, but more mature chloroplasts also have this capacity.

Plant proplastids (undifferentiated plastids) may differentiate into several forms, depending upon which function they perform in the cell, (see top graphic). They may develop into any of the following variants:[11]

    Chloroplasts: typically green plastids that perform photosynthesis.
        Etioplasts: precursors of chloroplasts.
    Chromoplasts: coloured plastids that synthesize and store pigments.
    Gerontoplasts: plastids that control the dismantling of the photosynthetic apparatus during plant senescence.
    Leucoplasts: colourless plastids that synthesize monoterpenes.

Leucoplasts differentiate into even more specialized plastids, such as:

    the aleuroplasts;
        Amyloplasts: storing starch and detecting gravity—for maintaining geotropism.
        Elaioplasts: storing fats.
        Proteinoplasts: storing and modifying protein.
    or Tannosomes: synthesizing and producing tannins and polyphenols.

Depending on their morphology and target function, plastids have the ability to differentiate or redifferentiate between these and other forms.
Plastomes and Chloroplast DNA/ RNA; plastid DNA and plastid nucleoids

Each plastid creates multiple copies of its own unique genome, or plastome, (from 'plastid genome')—which for a chlorophyll plastid (or chloroplast) is equivalent to a 'chloroplast genome', or a 'chloroplast DNA'.[12][13] The number of genome copies produced per plastid is variable, ranging from 1000 or more in rapidly dividing new cells, encompassing only a few plastids, down to 100 or less in mature cells, encompassing numerous plastids.

A plastome typically contains a genome that encodes transfer ribonucleic acids (tRNA)s and ribosomal ribonucleic acids (rRNAs). It also contains proteins involved in photosynthesis and plastid gene transcription and translation. But these proteins represent only a small fraction of the total protein set-up necessary to build and maintain any particular type of plastid. Nuclear genes (in the cell nucleus of a plant) encode the vast majority of plastid proteins; and the expression of nuclear and plastid genes is co-regulated to coordinate the development and differention of plastids.

Many plastids, particularly those responsible for photosynthesis, possess numerous internal membrane layers. Plastid DNA exists as protein-DNA complexes associated as localized regions within the plastid's inner envelope membrane; and these complexes are called 'plastid nucleoids'. Unlike the nucleus of a eukaryotic cell, a plastid nucleoid is not surrounded by a nuclear membrane. The region of each nucleoid may contain more than 10 copies of the plastid DNA.

Where the proplastid (undifferentiated plastid) contains a single nucleoid region located near the centre of the proplastid, the developing (or differentiating) plastid has many nucleoids localized at the periphery of the plastid and bound to the inner envelope membrane. During the development/ differentiation of proplastids to chloroplasts—and when plastids are differentiating from one type to another—nucleoids change in morphology, size, and location within the organelle. The remodelling of plastid nucleoids is believed to occur by modifications to the abundance of and the composition of nucleoid proteins.

In normal plant cells long thin protuberances called stromules sometimes form—extending from the plastid body into the cell cytosol while interconnecting several plastids. Proteins and smaller molecules can move around and through the stromules. Comparatively, in the laboratory, most cultured cells—which are large compared to normal plant cells—produce very long and abundant stromules that extend to the cell periphery.

In 2014, evidence was found of the possible loss of plastid genome in Rafflesia lagascae, a non-photosynthetic parasitic flowering plant, and in Polytomella, a genus of non-photosynthetic green algae. Extensive searches for plastid genes in both taxons yielded no results, but concluding that their plastomes are entirely missing is still disputed.[14] Some scientists argue that plastid genome loss is unlikely since even these non-photosynthetic plastids contain genes necessary to complete various biosynthetic pathways including heme biosynthesis.[14][15]

Even with any loss of plastid genome in Rafflesiaceae, the plastids still occur there as "shells" without DNA content,[16] which is reminiscent of hydrogenosomes in various organisms. """

comptime prompt_seven = """Elaioplasts are one of the three possible forms of leucoplasts, sometimes broadly referred to as such.[1] The main function of elaioplasts is synthesis and storage of fatty acids, terpenes, and other lipids, and they can be found in the embryonic leaves of certain plants, as well as the anthers of many flowering plants.[1][2][3][4]
Description

Like most leucoplasts, elaioplasts are non-pigmented organelles capable of alternating between the different forms of plastids. The elaioplast specifically is primarily responsible for the storage and metabolism of lipids,[5] among these roles, recent studies have shown that these organelles participate in the formation of terpenes and fatty acids.[2][3] Typically, they appear as small, rounded organelles filled by oil droplets.[1] Lipids found inside elaioplasts mirror those synthesized by prokaryotes, chiefly triacylglycerol and sterol esters, which cluster into the droplets visible by microscope.[1] As for their other components, elaioplasts also contain plastoglobuli associated proteins such as fibrillins, a protein family believed to be retained from the cyanobacterial ancestors of plastids.[4] Alongside the tapetosomes (clusters of oil and proteins produced by the endoplasmic reticulum), elaioplasts are frequently found in the tapetum of angiosperm anthers, where their products, oil from the plastid and protein from the tapetosome, are used to form the pollen coat of developing grains.[1] Following the maturation of pollen grains, these organelles are degraded and released into the anther loculus.[1] Found also in oilseeds, elaioplasts in this group provide lipids to be converted into carbohydrates which will serve as fuel in the embryo's germination.[4] Citrus specimens have been shown to have especially high amounts of elaioplasts in their fruit peels, where they are essential to the production of terpenes.[5]
Development

Within the plant, elaioplasts, as well as all other plastids, arise from proplastids in the dividing portion of the stem (meristem). These proplastids have not yet differentiated and, as such, can develop into any variety of known plastids, determined by the tissues they are present in.[6] In vegetative cells, proplastids usually follow a unidirectional pathway of development with no reversals between one form and the next. Reproductive cells, however, may have plastids that inter-convert frequently.[7] In the anthers of flowering plants, elaioplasts represent the final stage of plastid development within the tapetum, either emerging directly from proplastids or the conversion of other plastids, depending on the species and pollination strategy.[7]
Origin and inheritance

Plastids are hypothesized to have originated with an endosymbiotic event between an ancient eukaryote and cyanobacterial ancestor more than 1 billion years ago, where the bacteria was engulfed by the other and retained where it served as the metabolic center for photosynthesis.[8] Evidence of this can be observed today in the independent genomes characteristic of plastids, found to be closely related to modern cyanobacteria.[9] Since their ancient symbiotic event, the plastid genome has been reduced significantly, with the organelles themselves coding for around 100 of the 2500 associated proteins, everything else being transferred to the nuclear genome.[1]

Like most plastids, elaioplasts reproduce through binary fission independent from the division of the parent cell, a feature indicative of their bacterial ancestry.[1] This fission occurs just before cytokinesis, with the products then being transported to the daughter cells as a component of the cytoplasm.[1]

As a result of the ability to inter-convert between other types of the plastid family, elaioplasts share the same plastome(plastid genome) with all other plastids and are predominately inherited maternally in angiosperms.[5][7] As its name implies, maternal inheritance excludes the plastome of the father through one of two ways: during pollen development or in pollen tube formation.[7] During pollen development, paternal plastids are halted by microfilaments in the cytoskeleton just prior to microspore division or degeneration just after.[7] Paternal plastome contribution can also be prevented during pollen tube formation, where the plastids are separated from sperm cells as they fuse with the egg.[7] """

comptime prompt_eight = """The Roman aqueducts were a system of engineering structures built by the ancient Romans to transport water from distant sources into cities and towns. Constructed from a combination of stone, brick, and a special volcanic cement known as pozzolana, these channels supplied public baths, latrines, fountains, and private households across the empire. The water flowed largely by gravity alone, descending along a very gentle gradient maintained over distances that sometimes exceeded a hundred kilometres.

The earliest aqueduct in Rome, the Aqua Appia, was commissioned in 312 BC and ran almost entirely underground to protect it from contamination and enemy sabotage. As the city's population grew, later aqueducts such as the Aqua Marcia and the Aqua Claudia carried far greater volumes and rose onto towering arched bridges where the terrain dipped. At the height of the empire, eleven major aqueducts served the capital, together delivering an estimated one million cubic metres of water each day.

Beyond Rome itself, provincial cities throughout Gaul, Hispania, and North Africa built their own aqueducts, many of which still stand today. The Pont du Gard in southern France and the aqueduct of Segovia in Spain remain among the best preserved, their multi-tiered arches a testament to the durability of Roman construction. Maintenance was the responsibility of a dedicated office, and a permanent staff of workers inspected the channels, cleared sediment, and repaired leaks.

The decline of the aqueduct network paralleled the broader collapse of Roman administrative power in the West. As central authority weakened, the resources and expertise needed to maintain the channels disappeared, and many fell into disrepair or were deliberately cut during sieges. Nevertheless, the underlying principles of gradient flow and durable masonry influenced water engineering for centuries, and several aqueducts were restored and returned to service during the Renaissance."""


def sample_prompts() -> List[String]:
    var prompts = List[String](capacity=8)
    prompts.append(String(prompt_one))
    prompts.append(String(prompt_two))
    prompts.append(String(prompt_three))
    prompts.append(String(prompt_four))
    prompts.append(String(prompt_five))
    prompts.append(String(prompt_six))
    prompts.append(String(prompt_seven))
    prompts.append(String(prompt_eight))
    return prompts^
