


GB=1024**3  #bytes in 1GB
double=8    #bytes per double precision


def storageFull(N=10,D=1):
    return (N*D)**2*double/GB

def storageDecomposition(N=10,D=1,cg=False):
    #  WX: DxN
    #  Kp: NxN
    # Kpp: NxN
    #   W: ignored (1,DxD)
    if cg:
        total=3*D*N + 3*N**2
    else:
        total=D*N +2*N**2

    return total*double/GB

def printStorage(N=10,D=1,cg=False):
    full=storageFull(N,D)
    decomp=storageDecomposition(N,D,cg)
    s=f"(D,N)=({D:3d},{N:5d}):" 
    for tot in [full,decomp]:
        if tot<1:
            s+=f" {tot*1024:7.2f} MB"
        elif tot>1000:
            s+=f" {tot/1024:7.2f} TB"
        else:
            s+=f" {tot:7.2f} GB"
    print(s)
    # print(f"(N,D)=({N:5d},{D:5d}): {full} | {decomp}")




if __name__ == '__main__':
    # cg=True
    cg=False
    for D in [2,3,10,25,100,250]:
        for N in [100,10**3, 5*10**3, 10**4, 2*10**4]:
            printStorage(N,D,cg)