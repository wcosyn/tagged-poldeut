c Reference table for taggedpd.christy_bosted (port of F1F209.f).
c Columns: W Q2 | christy507: F1p R sigT sigL | resmodd: F1avg |
c   f1f2in09 proton F1 F2 R | neutron F1 F2 R | deuteron F1 F2 R
c Build: see README in this directory.
      program cbref
      implicit none
      real*8 w,w2,q2,qs(13),f1,r,st,sl,favg
      real*8 f1p,f2p,rp,f1n,f2n,rn,f1d,f2d,rd
      real*8 xvald0(50)
      integer i,j
      data qs/0.d0,1.d-5,1.d-4,1.d-3,1.d-2,5.d-2,0.1d0,0.3d0,0.6d0,
     >        1.d0,2.d0,4.d0,8.d0/
      data xvald0/
     >  0.1964E+01, 0.1086E+01, 0.5313E-02, 0.1265E+01, 0.8000E+01,
     >  0.2979E+00, 0.1354E+00, 0.2200E+00, 0.8296E-01, 0.9578E-01,
     >  0.1094E+00, 0.3794E+00, 0.8122E+01, 0.5189E+01, 0.3290E+01,
     >  0.1870E+01, 0.6110E+01,-0.3464E+02, 0.9000E+03, 0.1717E+01,
     >  0.4335E-01, 0.1915E+03, 0.2232E+00, 0.2119E+01, 0.2088E+01,
     > -0.3029E+00, 0.2012E+00, 0.1104E-02, 0.2276E-01,-0.4562E+00,
     >  0.2397E+00, 0.1204E+01, 0.2321E-01, 0.5419E+03, 0.2247E+00,
     >  0.2168E+01, 0.2266E+03, 0.7649E-01, 0.1457E+01, 0.1318E+00,
     > -0.7534E+02, 0.1776E+00, 0.1636E+01, 0.1350E+00,-0.5596E-02,
     >  0.5883E-02, 0.1934E+01, 0.3800E+00, 0.3319E+01, 0.1446E+00/
      do j=1,13
       q2 = qs(j)
       do i=0,225
        w = 0.9503d0 + 0.01d0*i
        w2 = w*w
        f1=0.d0; r=0.d0; st=0.d0; sl=0.d0; favg=0.d0
        if(w2.gt.1.155d0) then
          call christy507(w2,q2,f1,r,st,sl)
          call resmodd(w2,q2,xvald0,favg)
        endif
        call f1f2in09(1.d0,1.d0,q2,w2,f1p,f2p,rp)
        call f1f2in09(0.d0,1.d0,q2,w2,f1n,f2n,rn)
        call f1f2in09(1.d0,2.d0,q2,w2,f1d,f2d,rd)
        if(w2.le.1.155d0) then
          rp=0.d0; rn=0.d0
        endif
        write(6,'(16es24.16)') w,q2,f1,r,st,sl,favg,
     >    f1p,f2p,rp,f1n,f2n,rn,f1d,f2d,rd
       enddo
      enddo
      end
