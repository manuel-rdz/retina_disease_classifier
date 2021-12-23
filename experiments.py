#import timm

#print(timm.list_models('*it*', pretrained=True))

nums = [-1,1,2,3,4,5,6,7]


if (-1 in nums) and (sorted(nums)[1:]) in {list(range(2, len(nums))), list(range(1, len(nums)-1))}:
    pass