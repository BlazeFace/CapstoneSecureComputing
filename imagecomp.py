from ssim3 import *

values = 0
# generates the values for each jpg images after video processing is complete
for i in range(655):
    values += main("output/test/frame%d.jpg" % i, "output/test_scaled/frame%d.jpg" % i)

#returns average SSIM value of all images
print(values / 655)


#main("example_restored.jpg", "example_restored.jpg")

#old code not neccessary to read
'''
def load(x1, x2):
    arr1 = [0] * len(x1)
    arr2 = [0] * len(x2)
    count1 = 0
    count2 = 0

    for i in x1:
        if (torch.is_tensor(i)):
            i = i.detach().cpu().numpy()
        arr1[count1] = Image.fromarray(i.astype(np.uint8)).convert('RGB')
        count1 = count1 + 1
    
    for j in x2:
        if (torch.is_tensor(j)):
            j = j.detach().cpu().numpy()
        arr2[count2] = Image.fromarray(j.astype(np.uint8)).convert('RGB')
        count2 = count2 + 1

def sorter(arr1, arr2):
    print(5)
'''

'''
def image_comp(filename1, filename2):
    i1 = Image.open("output.jpeg")
    i2 = Image.open("out2.jpg")
    assert i1.mode == i2.mode, "Different kinds of images."
    assert i1.size == i2.size, "Different sizes."
    
    pairs = zip(i1.getdata(), i2.getdata())
    if len(i1.getbands()) == 1:
        # for gray-scale jpegs
        dif = sum(abs(p1-p2) for p1,p2 in pairs)
    else:
        dif = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))
    
    ncomponents = i1.size[0] * i1.size[1] * 3
    print ("Difference (percentage):", (dif / 255.0 * 100) / ncomponents)
'''
