import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    output = dict()
    """Get all the pages that exists in the corpus"""
    all_pages = corpus.keys()
    """Set each page with probability of 0 at start"""
    for each_page in all_pages:
        output[each_page] = 0

    """Get all possible path from the current page"""
    possible_pages = corpus.get(page)
    # print(possible_pages)

    """If there are possible pages"""
    if len(possible_pages) > 0:
        """Distribute the prob 0.85 between all possible pages"""
        possible_pages_prob = damping_factor / len(possible_pages)

        """Add that prob to each possible page value in the output dict"""
        for each_page in possible_pages:
            output[each_page] += possible_pages_prob

    
        """Distribute the prob 0.15 between all pages"""
        random_page_prob = (1 - damping_factor) / len(all_pages)

        """Add that prob to each page value in the output dict"""
        for a_page in all_pages:
            output[a_page] += random_page_prob

    else:
        """Distribute prob 1.0 between all pages"""
        all_pages_prob = 1 / len(all_pages)

        """Add that prob to each page value in the output dict"""
        for each_page in all_pages:
            output[each_page] += all_pages_prob
    

    return output


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    """The object that the function has to return"""
    pagerank = dict()
    
    """Counter for keep track of the samples generated"""
    counter = 0
    
    """Variable to keep track of sample each page"""
    next_page = None

    """List to save and keep track of all samples"""
    samples_list = list()

    """This code will be executed n times adding sample to samples list"""
    while counter < n:
        counter += 1
        
        if next_page is None:
            """This gets randomly a page from the corpus"""
            random_page = random.choice(list(corpus.keys()))

            """The inital sample to work next others"""
            first_sample = transition_model(corpus,random_page,damping_factor)

            """Add first sample to samples list"""
            samples_list.append(first_sample)

            """Set the next page for the next sample"""
            next_page = get_next_page(first_sample)

        else:
            """Create a sample"""
            nsample = transition_model(corpus,next_page,damping_factor)

            """Add that sample to the samples list"""
            samples_list.append(nsample)

            """Update the next page for the next sample"""
            next_page = get_next_page(nsample)
    
    # print(samples_list)
    """Quantity of pages in corpus"""
    qpages = len(list(corpus.keys()))
    # print(qpages)
    total = 0
    # sample_tracker = list()
    weights_sum = list()
    # weights_len = 0
    
    """
    Set values 0 in positions page index where the weights of this page has to be sum
    The index of each value depends of the quantity of pages in corpus
    """
    for page in range(qpages):
        weights_sum.insert(page, 0)

    """Loop all the sample we have save in samples list"""
    for sample in samples_list:
        # sample_pages = list(sample.keys())
        """Getting the probability of each page in a list"""
        sample_weights = list(sample.values())
        
        # weights_len += 1

        """
        This sums all the weights of each page in different position
        e.g. [total_weight_page_1, total_weight_page_2, total_weight_page_3]
        """
        for page, weight in enumerate(sample_weights):
            weights_sum[page] += weight

    # print(weights_sum)
    # print(weights_len)
    
    """
    Dividing each weight of each page by the 
    damping factor to get the average
    Dividing each "total_weight_page_n"
    """
    for page in range(qpages):
        weights_sum[page] = weights_sum[page] / n
        total += weights_sum[page]

    # print(weights_sum)
    # print("total", total)

    """Join all the pages in the corpus with their weights"""
    for page, weight in zip(list(corpus.keys()),weights_sum):
        # print(page, weight)

        """Add the page to the pagerank with its weights as value"""
        pagerank[page] = weight

    # print(pagerank)
    """Return the final pagerank results"""
    return pagerank


def get_next_page(sample):
    sample_pages = list(sample.keys())
    sample_weights = list(sample.values())
    next_page = random.choices(sample_pages,sample_weights, k=1)
    return next_page[0]


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
