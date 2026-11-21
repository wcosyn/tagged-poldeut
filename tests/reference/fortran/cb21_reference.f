c Reference table for taggedpd.christy_bosted21 (port of F1F2IN21_v1.0.f,
c M. E. Christy, JLab SIMC; PRELIMINARY fit). Run in a directory containing
c f1f2tables/ (on- and off-shell CD-Bonn tables from SIMC).
c Columns: W Q2 | rescsp sigT sigL | rescsn sigT sigL |
c   F1F2IN21 proton F1 F2 | neutron F1 F2 | deuteron F1 F2
      program cb21ref
      implicit none
      real*8 w,w2,q2,qs(16),stp,slp,stn,sln
      real*8 f1p,f2p,f1n,f2n,f1d,f2d,f1,f2,fl
      integer i,j,wfn
      logical first
      data qs/0.d0,1.d-5,1.d-4,1.d-3,1.d-2,4.9d-2,5.d-2,0.1d0,0.3d0,
     >        0.6d0,1.d0,2.d0,4.d0,8.d0,15.d0,25.d0/
      first = .true.
      wfn = 2
      call sqesub(1.d0,1.d0,wfn,f2,f1,fl,first)
      do j=1,16
       q2 = qs(j)
       do i=0,225
        w = 0.9503d0 + 0.01d0*i
        w2 = w*w
        call rescsp(w2,q2,stp,slp)
        call rescsn(w2,q2,stn,sln)
        call f1f2in21(1.d0,1.d0,q2,w2,f1p,f2p)
        call f1f2in21(0.d0,1.d0,q2,w2,f1n,f2n)
        call f1f2in21(1.d0,2.d0,q2,w2,f1d,f2d)
        write(6,'(12es24.16)') w,q2,stp,slp,stn,sln,
     >    f1p,f2p,f1n,f2n,f1d,f2d
       enddo
      enddo
      end
