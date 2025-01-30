#definetly overcomplicated
import random
sizes = ["tiny", "minuscule", "puny", "undersized", "microscopic", "scanty", "meagre", "diminutive", "paltry", "insignificant"]
quantities = ["insufficient", "scarce", "sparse", "limited", "deficient", "inadequate", "paltry", "negligible", "depleted", "lacking", "diminished", "minimal", "short", "restricted"]
qualities = ["arrogant", "callous", "deceitful", "greedy", "hostile", "lazy", "obnoxious", "rude", "selfish", "vindictive"]
ages = ["decrepit", "senile", "doddering", "over-the-hill", "ancient", "geriatric", "past-it", "old-fashioned", "fossilized", "wizened"]
shapes = ["bulky", "clunky", "crooked", "lopsided", "misshapen", "stubby", "twisted", "warped", "awkward", "bent"]
colors = ["dull", "drab", "muddy", "bleak", "harsh", "washed-out", "garish", "brash", "loud", "somber"]
adjectives=["sizes","qualities","qualities","ages","shapes","colors"]

def generate_adjectives():
    chosen_adj=[]
    already_chosen=[]
    loop=random.randint(2,4)
    while loop !=0:
        choice=random.choice(adjectives)
        if choice  not in already_chosen:
            match(choice):
                case "sizes":
                    chosen_adj.append([random.choice(sizes),loop])
                    already_chosen.append("sizes")

                case "qualities":
                    chosen_adj.append([random.choice(quantities),loop])
                    already_chosen.append("qualities")

                case "quantities":
                    chosen_adj.append([random.choice(qualities),loop])
                    already_chosen.append("quantities")

                case "ages":
                    chosen_adj.append([random.choice(ages),loop])
                    already_chosen.append("ages")

                case "shapes":
                    chosen_adj.append([random.choice(shapes),loop])
                    already_chosen.append("shapes")

                case "colors":
                    chosen_adj.append([random.choice(colors),loop])
                    already_chosen.append("colors")

        loop-=1
    return chosen_adj

def order_adj(unordered):
    sorted_list = sorted(unordered, key=lambda x: x[1], reverse=True)
    output=[]
    for adj in sorted_list:
        output.append(adj[0])
    return output
        

print("Josh is a",*order_adj(generate_adjectives()), "person")