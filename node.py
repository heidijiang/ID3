class Node:
    def __init__(self):
        self.label = None
        self.name = None
        self.value = None
        self.children = {}
	# you may want to add additional fields here...
    def print_tree(self,space = '-'):
        space = space + '--'
        for child in self.children:
            if self.label is None:
                num,denom,classes = Node.sum_tree(self.children[child])
                print(space + self.name + ': ' + str(child) + '=' +
                      str(num) + '/' + str(denom) + ', ' + 
                         str(classes) + ' ' + str(float(num)/float(denom)))
                classes.clear()
                Node.print_tree(self.children[child],space)

    def sum_tree(self,num=0,denom=0,classes=dict()):
        
        if self.label is not None:
            for i in self.value:
                denom+=1
                if i['Class']==self.label:
                    num+=1
                if classes.get(i['Class']) is None:
                    classes[i['Class']] = 1
                else:
                    classes[i['Class']] += 1
                        
                
        else:
            for child in self.children:
                num,denom,classes = Node.sum_tree(self.children[child],num,denom)
        return num,denom,classes
    
    def get_node_num(self,count=1):
        if self.label is not None:
            pass
#            count+=1
#            return count
        else:
            for child in self.children:
                count+=1
                count = Node.get_node_num(self.children[child],count)
        return count
                
#        