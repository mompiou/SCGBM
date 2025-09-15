from __future__ import division
import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from PIL import Image
#import PngImagePlugin
import sys
import os
from fractions import Fraction

pi=np.pi


    
###################################################################"
##### Fonction rotation 
####################################################################

def rotation(phi1,phi,phi2):
   phi1=phi1*pi/180;
   phi=phi*pi/180;
   phi2=phi2*pi/180;
   R=np.array([[np.cos(phi1)*np.cos(phi2)-np.cos(phi)*np.sin(phi1)*np.sin(phi2),
            -np.cos(phi)*np.cos(phi2)*np.sin(phi1)-np.cos(phi1)*
            np.sin(phi2),np.sin(phi)*np.sin(phi1)],[np.cos(phi2)*np.sin(phi1)
            +np.cos(phi)*np.cos(phi1)*np.sin(phi2),np.cos(phi)*np.cos(phi1)
            *np.cos(phi2)-np.sin(phi1)*np.sin(phi2), -np.cos(phi1)*np.sin(phi)],
            [np.sin(phi)*np.sin(phi2), np.cos(phi2)*np.sin(phi), np.cos(phi)]],float)
   return R

####################################################################
##### Fonction rotation autour d'un axe 
####################################################################

def Rot(th,a,b,c):
   th=th*pi/180;
   aa=a/np.linalg.norm([a,b,c]);
   bb=b/np.linalg.norm([a,b,c]);
   cc=c/np.linalg.norm([a,b,c]);
   c1=np.array([[1,0,0],[0,1,0],[0,0,1]],float)
   c2=np.array([[aa**2,aa*bb,aa*cc],[bb*aa,bb**2,bb*cc],[cc*aa,
                cc*bb,cc**2]],float)
   c3=np.array([[0,-cc,bb],[cc,0,-aa],[-bb,aa,0]],float)
   R=np.cos(th)*c1+(1-np.cos(th))*c2+np.sin(th)*c3

   return R    





####################################################################
##### Fonction desorientation
####################################################################        
def Rota(t,u,v,w,g):
    Ae=np.dot(g,np.array([u,v,w]))
    Re=Rot(t,Ae[0],Ae[1],Ae[2])
    return Re

def cryststruct():
    global cs

    if  gam==90 and alp==90 and bet==90 and a==b and b==c:
       cs=1

    if gam==120 and alp==90 and bet==90:
        cs=2

    if gam==90 and alp==90 and bet==90 and a==b and b!=c: 
        cs=3  


    if alp!=90 and a==b and b==c:
        cs=4  

    if gam==90 and alp==90 and bet==90 and a!=b and b!=c:
        cs=5  
    
    if gam!=90 and alp==90 and bet==90 and a!=b and b!=c:
        cs=6  

    if gam!=90 and alp!=90 and bet!=90 and a!=b and b!=c: 
        cs=7  
    return cs
    
def Sy(g):
    global cs
    if cs==1:
        S1=Rota(90,1,0,0,g);
        S2=Rota(180,1,0,0,g);
        S3=Rota(270,1,0,0,g);
        S4=Rota(90,0,1,0,g);
        S5=Rota(180,0,1,0,g);
        S6=Rota(270,0,1,0,g);
        S7=Rota(90,0,0,1,g);
        S8=Rota(180,0,0,1,g);
        S9=Rota(270,0,0,1,g);
        S10=Rota(180,1,1,0,g);
        S11=Rota(180,1,0,1,g);
        
        S12=Rota(180,0,1,1,g);
        S13=Rota(180,-1,1,0,g);
        S14=Rota(180,-1,0,1,g);
        S15=Rota(180,0,-1,1,g);
        S16=Rota(120,1,1,1,g);
        S17=Rota(240,1,1,1,g);
        S18=Rota(120,-1,1,1,g);
        S19=Rota(240,-1,1,1,g);
        S20=Rota(120,1,-1,1,g);
        S21=Rota(240,1,-1,1,g);
        S22=Rota(120,1,1,-1,g);
        S23=Rota(240,1,1,-1,g);
        S24=np.eye(3,3);
        S=np.vstack((S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20,S21,S22,S23,S24))
        
  
    
    if cs==2:
        S1=Rota(60,0,0,1,g);
        S2=Rota(120,0,0,1,g);
        S3=Rota(180,0,0,1,g);
        S4=Rota(240,0,0,1,g);
        S5=Rota(300,0,0,1,g);
        S6=np.eye(3,3);
        S7=Rota(180,0,0,1,g);
        S8=Rota(180,0,1,0,g);
        S9=Rota(180,1/2,np.sqrt(3)/2,0,g);
        S10=Rota(180,-1/2,np.sqrt(3)/2,0,g);
        S11=Rota(180,np.sqrt(3)/2,1/2,0,g);
        S12=Rota(180,-np.sqrt(3)/2,1/2,0,g);
        S=np.vstack((S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12))
        
       
        
    if cs==3:
        S1=Rota(90,0,0,1,g);
        S2=Rota(180,0,0,1,g);
        S3=Rota(270,0,0,1,g);
        S4=Rota(180,0,1,0,g);
        S5=Rota(180,1,0,0,g);
        S6=Rota(180,1,1,0,g);
        S7=Rota(180,1,-1,0,g);
        S8=np.eye(3,3)
        S=np.vstack((S1,S2,S3,S4,S5,S6,S7,S8))
        
        
      
        
    if cs==4:
        S1=Rota(60,0,0,1,g);
        S2=Rota(120,0,0,1,g);
        S3=Rota(180,0,0,1,g);
        S4=Rota(240,0,0,1,g);
        S5=Rota(300,0,0,1,g);
        S6=np.eye(3,3);
        S7=Rota(180,0,0,1,g);
        S8=Rota(180,0,1,0,g);
        S9=Rota(180,1/2,np.sqrt(3)/2,0,g);
        S10=Rota(180,-1/2,np.sqrt(3)/2,0,g);
        S11=Rota(180,np.sqrt(3)/2,1/2,0,g);
        S12=Rota(180,-np.sqrt(3)/2,1/2,0,g);
        S=np.vstack((S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12))
        
    
              
         
    if cs==5:
        S1=Rota(180,0,0,1,g);
        S2=Rota(180,1,0,0,g);
        S3=Rota(180,0,1,0,g);
        S4=np.eye(3,3);
        S=np.vstack((S1,S2,S3,S4))
        
                
        
    if cs==6:
        S1=Rota(180,0,1,0,g);
        S2=np.eye(3,3);
        S=np.vstack((S1,S2))
        
 
        
    if cs==7:
        S=np.eye(3,3);
        
    
    return S
    
def null(A, rcond=None):

    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q
    

def desorientation(phi1a,phia,phi2a,phi1b,phib,phi2b):
    global D0,S,D1,cs,V,alp,bet,gam
    
    #alp=alp*pi/180;
    #bet=bet*pi/180;
    #gam=gam*pi/180;
    

    gA=rotation(phi1a,phia,phi2a)
    gB=rotation(phi1b,phib,phi2b)
    k=0
    S=Sy(gA)
    
    D0=np.zeros((S.shape[0]//3,5))
    D1=np.zeros((S.shape[0]//3,3))
    
    for i in range(0,S.shape[0],3):
        In=np.dot(np.array([[S[i,0],S[i+1,0],S[i+2,0]],[S[i,1],S[i+1,1],S[i+2,1]],[S[i,2],S[i+1,2],S[i+2,2]]]),gA)
        Ing=np.dot(In,np.array([0,0,1]))
        In2=np.dot(Rot(-phi2b,Ing[0],Ing[1],Ing[2]),In)
        Ing2=np.dot(In2,np.array([1,0,0]))
        In3=np.dot(Rot(-phib,Ing2[0],Ing2[1],Ing2[2]),In2)
        Ing3=np.dot(In3,np.array([0,0,1]))
        A=np.dot(Rot(-phi1b,Ing3[0],Ing3[1],Ing3[2]),In3)-np.eye(3)
        V=null(A,0.001).T

        
        if 0.5*(np.trace(A+np.eye(3))-1)>1:
        	D0[k,3]=0
        elif 0.5*(np.trace(A+np.eye(3))-1)<-1:
        	D0[k,3]=180
        else:
        	D0[k,3]=np.arccos(0.5*(np.trace(A+np.eye(3))-1))*180/pi
        	
        if np.abs(D0[k,3])<1e-5:
            D0[k,0]=0
            D0[k,1]=0
            D0[k,2]=0
        else:
                D0[k,0]=V[0,0]/np.linalg.norm(V)
                D0[k,1]=V[0,1]/np.linalg.norm(V)
                D0[k,2]=V[0,2]/np.linalg.norm(V)
	       
        Ds1=np.dot(np.linalg.inv(gB),np.array([D0[k,0],D0[k,1],D0[k,2]]))

        F0=Fraction(Ds1[0]).limit_denominator(10)
        F1=Fraction(Ds1[1]).limit_denominator(10)
        F2=Fraction(Ds1[2]).limit_denominator(10)
                    
        D1[k,0]=F0.numerator*F1.denominator*F2.denominator
        D1[k,1]=F1.numerator*F0.denominator*F2.denominator
        D1[k,2]=F2.numerator*F0.denominator*F1.denominator
		   
		
		
        if D0[k,2]<0:
                D0[k,0]=-D0[k,0]
                D0[k,1]=-D0[k,1]
                D0[k,2]=-D0[k,2]
                D1[k,0]=-D1[k,0]
                D1[k,1]=-D1[k,1]
                D1[k,2]=-D1[k,2]
           
       
       
        D0[k,4]=k


        k=k+1
    
    for i in range (0,len(D0)):
        ang=float(anglenorm(phi1b,phib,phi2b,D1[i,0],D1[i,1],D1[i,2]))
        if ang<10 and D0[i,3]>175:
            D0[i,3]=0.0
            
           
    ii=np.nanargmin(np.abs(D0[:,3]))
        

    return D1[ii,:],D0[ii,3]

a=4.0496
b=4.0496
c=4.0496
alp=90
bet=90
gam=90
j=0


def anglenorm(phi1,phi,phi2,e1,e2,e3):
    
    alp=90*pi/180;
    bet=90*pi/180;
    gam=90*pi/180;
    W=a*b*c*np.sqrt(1-(np.cos(alp)**2)-(np.cos(bet))**2-(np.cos(gam))**2+2*np.cos(alp)*np.cos(bet)*np.cos(gam))
    D=np.array([[a,b*np.cos(gam),c*np.cos(bet)],[0,b*np.sin(gam),  c*(np.cos(alp)-np.cos(bet)*np.cos(gam))/np.sin(gam)],[0,0,W/(a*b*np.sin(gam))]])
    Dstar=np.transpose(np.linalg.inv(D))
    c1=np.array([e1,e2,e3])
    c2=np.array([0,0,1]) 
    c1c=np.dot(Dstar,c1)
    c2c=np.dot(Dstar,c2)
    rota=rotation(phi1,phi,phi2)
    rol=np.linalg.inv(rotation(phi1,phi,phi2))
    c2c=np.dot(rol,c2c)
    the=np.arccos(np.dot(c1c,c2c)/(np.linalg.norm(c1c)*np.linalg.norm(c2c)))                   
    thes=str(np.around(the*180/pi,decimals=2))

    return thes


   
def mainone():
    
    sortie=open("rotation.txt","w")
    sortie2=open("color.txt","w")
    h=0
    desor_in = np.loadtxt("angles.txt")
    cryststruct()
    while h<(len(desor_in)):
        phi1a=desor_in[h,1]
        phia=desor_in[h,2]
        phi2a=desor_in[h,3]
        phi1b=desor_in[h,4]
        phib=desor_in[h,5]
        phi2b=desor_in[h,6]
        axes,angle=desorientation(phi1a,phia,phi2a,phi1b,phib,phi2b)
        coul=angle*255/90
        sortie.write(str(axes[0])+', '+str(axes[1])+', '+str(axes[2])+', '+str(angle) + '\n')
        sortie2.write(str(coul)+','+'\n')
        h=h+1
    sortie.close()
    sortie2.close()


#cryststruct()
#phi1a=272
#phia=37.9
#phi2a=86.85
#phi1b=268.056
#phib=40.623
#phi2b=89.961
#e1=50   
#e2=0
#e3=49

#ang=anglenorm(phi1b,phib,phi2b,e1,e2,e3)
#print ang

#des=desorientation(phi1a,phia,phi2a,phi1b,phib,phi2b)



mainone()    

              
              
              
