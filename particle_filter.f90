!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      module ldu_solver
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      contains
!-----------------------------------------------------------------------
       subroutine gauss_jordan(cvrc, x, L2, n, sstep)
!-----------------------------------------------------------------------
      implicit none
      integer, parameter         ::  rkind=selected_real_kind(15,307)
      integer, intent(in)        ::  n, sstep
      integer                    ::  ii, i, k, m 
      real(rkind), intent(in)    ::  cvrc(n, n), L2(n, sstep)
      real(rkind)                ::  a(n, n)
      real(rkind)                ::  am, ar, t
      real(rkind), intent(out)   ::  x(n, sstep)
!-----------------------------------------------------------------------

      do ii = 1, sstep

        a(:, :)  = cvrc(:, :)
        x(:, ii) = L2(:, ii)

          do k = 1, n

          if(am == 0.0d0) stop "A is singular !"

            ar          = 1.0d0 / a(k, k)
            a(k, k)     = 1.0d0
            a(k, k+1:n) = ar * a(k, k+1:n)
            x(k, ii)    = ar * x(k, ii)

           do i = 1, n
            if(i /= k) then
              a(i, k+1:n)   = a(i, k+1:n) - a(i, k) * a(k, k+1:n)
              x(i, ii)      = x(i, ii) - a(i, k) * x(k, ii)
              a(i, k)       = 0.0d0
            end if
           end do

          end do

      end do


      end subroutine gauss_jordan

!-----------------------------------------------------------------------

      end module ldu_solver

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


! Å™module program
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
! Å´main program


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      program particle_filter
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      use ldu_solver
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      implicit none
      integer, parameter :: rkind=selected_real_kind(15,307)
      integer            :: nsample, nprm, sstep, obs_number, &
                            sample_step, int_k
      integer            :: i, ii, k, kk, kkk, j, jj
      integer            :: idummy

      real(rkind), allocatable :: PRM(:, :), APi(:, :), AP_i(:, :)
      real(rkind), allocatable :: y(:, :), cvrc(:, :), xn(:, :), h(:, :)
      real(rkind), allocatable :: L(:, :), L_1(:, :), L_2(:, :), &
                                  L_3(:, :), L_i(:,:)
      real(rkind)              :: L_4, sum_L, L_P
      real(rkind), allocatable :: L_t(:, :), L_x(:), L_sum_t(:, :)

!cccc Definitions of variables-1 cccccccccccccccccccccccccccccccccccccccc
!   PRM      :  particle
!   AP       :  identified parameter
!   nsample  :  the number of particle
!   nprm     :  the number of identified parameter
!   h        :  observation_matrix
!   y        :  bservation_vector
!   xn       :  Monte Calro Simulation
!   xnd      :  dummy of xn
!   cvrc     :  variance co-variance Matrix
!   L        :  likelihood(exp(0.5Å~((yn-hn•xn)É∞^-1(yn-hn•xn))))
!   L_1      :  hn•xn , parameter-1
!   L_2      :  yn-hn•xn , parameter-2
!   L_3      :  É∞^-1(yn-hn•xn) , parameter-3
!   L_4      :  (yn-hn•xn)É∞^-1(yn-hn•xn) , parameter-4
!   L_i      :  parametere of likelihood
!   L_t      :  likelihood at t(time)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      open(10,  file = 'obsdata.dat')               !open observation data file
      open(20,  file = 'co-variance_matrix.txt')    !open variance co-variance matrix file
      open(30,  file = 'obs_matrix.txt')            !open observation matrix file
      open(90,  file = 'particle.txt')              !open generated particle file
      open(100, file = 'control_parameter.txt')     !open control parameter file
      open(120, file = 'IP.txt')                    !make identidfied paraemters file

      read(100, *) sstep                            !read step of direct analysis
      read(100, *) nsample                          !read the number of particle
      read(100, *) obs_number                       !read the number of observation point
      read(100, *) nprm                             !read the number of identified parameter
      close(100)


!cccc Definitions of variables-2 ccccccccccccccccccccccccccccccccccccccc
!  sstep         :  the number of simulation step
!  obs_number    :  the number of observation points
!  cvrc          :  variance-covariance matrix
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      allocate(L(sstep, nsample), PRM(nsample, nprm))
      allocate(y(obs_number, sstep), cvrc(obs_number, obs_number))
      allocate(h(obs_number, obs_number))

      read(90,  *) ((PRM(i, j), j = 1, nprm), i=1, nsample)
      close(90)


!----- Read observation data -----
      read(10,  *) ((y(i, j), i = 1, obs_number), j=1,sstep)
      close(10)


!----- Read co-variance matrix(cvrc) -----
      read(20, *) ((cvrc(i, j),j=1, obs_number), i=1, obs_number)
      close(20)


!----- Read observation matrix(h) -----
      read(30, *) ((h(i, j), j = 1, obs_number), i=1, obs_number)
      close(30)


!ccccccc START LOOP cccccccccccccccccccccccccccccccccccccccccccccccccccc

      loop_timestep : do sample_step = 1, nsample

      open(1000, file = 'MCS.dat')

!----------------------------------------------------------------------

      allocate(L_1(obs_number, sstep), L_2(obs_number, sstep),&
               L_3(obs_number, sstep), L_i(sstep, 1),&
               xn(obs_number, sstep))

!----------------------------------------------------------------------

      L_1 = 0._rkind
      L_2 = 0._rkind

!ccc Read state vector(xn) ccccccccccccccccccccccccccccccccccccccccccccc

      read(1000, *) idummy
      read(1000, *) ((xn(i, j), i = 1, obs_number),j = 1, sstep)

!ccc Make L1-matrix(hn•xn)([call dgemm(L1,h,xn)]) ccccccccccccccccccccc

      do i = 1, obs_number
       do j = 1, sstep
        L_1(i, j) = 0.d0
        do k = 1, obs_number
         L_1(i, j) = L_1(i, j) + h(i, k) * xn(k, j)
        end do
       end do
      end do


!ccccc Make L2-matrix(yn-hn•xn) cccccccccccccccccccccccccccccccccccccccc

      L_2(1 : obs_number, 1 : sstep) = y(1 : obs_number, 1 : sstep)&
                                     - L_1(1 : obs_number, 1 : sstep)


!ccccc Make L3-matrix(É∞^-1(yn-hn•xn)) (inverse matrix) cccccccccccccccc

       call gauss_jordan(cvrc, L_3, L_2, obs_number, sstep)


!ccccc Compute L( exp(0.5Å~((yn-hn•xn)É∞^-1(yn-hn•xn))) ) cccccccccccccc

!    L_i : dummy parameter of L

      L_i = 0._rkind
       do i = 1, sstep
        L_4 = dot_product(L_2(1:obs_number, i), L_3(1:obs_number, i))
        L_i(i, 1) = exp(-0.5_rkind*L_4)
       end do

      L(1:sstep, sample_step) = L_i(1:sstep, 1)


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      deallocate(xn, L_1, L_2, L_3, L_i)

      end do loop_timestep

      deallocate(y, cvrc)


!ccc  Regularization of weight  cccccccccccccccccccccccccccccccccccccccc

      do j = 1, sstep

       sum_L = 0._rkind

       do i = 1, nsample 
        sum_L = sum_L + L(j,i)
       end do

       do ii = 1, nsample
        L(j, ii) = L(j, ii) / sum_L
       end do

      end do


!ccc  Make weight for each time step  cccccccccccccccccccccccccccccccccc

      allocate(L_t(sstep, nsample), L_x(nsample) ,L_sum_t(sstep, nsample))

      L_t = 0._rkind

      kk = sstep - 1
      L_t(1, 1:nsample) = L(1, 1:nsample)

      do i = 1, kk

      L_P = 0._rkind

        k = i + 1

       if (i == 1)then

         do j = 1,nsample
          L_x(j) = L(i, j) * L(k, j)
          L_P = L_x(j) + L_P
         end do

        else

         do j = 1, nsample
          L_x(j) = L_x(j) * L(k, j)
          L_P = L_x(j) + L_P
         end do

        end if

        do j = 1, nsample
         L_x(j) = L_x(j) / L_P
        end do

        kkk = i + 1
        L_t(kkk, 1:nsample) = L_x(1:nsample)

      end do


!ccc output weitht ccccccccccccccccccccccccccccccccccccccccccccccccccccc

      open(10001, file = 'weight.txt')

      do i = 1, nsample
        write(10001, *) (L_t(j, i), j = 1, sstep)
      end do

      close(10001)


!ccc  Makeidentified parameters  ccccccccccccccccccccccccccccccccccccccc

      allocate(APi(sstep, nprm), AP_i(nsample, nprm))

      do i = 1, sstep
       do k = 1, nprm
        AP_i  =0._rkind
        do j = 1, nsample
         jj = j-1
          if(j == 1) then
           AP_i(j, k) = L_t(i, j) * PRM(j, k)
          else
           AP_i(j, k) = L_t(i, j) * PRM(j, k)
           AP_i(j, k) = AP_i(j, k)+AP_i(jj, k)
          end if
         end do
        APi(i, k)=AP_i(nsample, k)
       end do
      end do

      do i = 1, sstep
       write(120, '(10e15.7)') (APi(i, k), k = 1, nprm)
      end do

!-----------------------------------------------------------------------

      deallocate(L, L_t, L_x, L_sum_t)
      deallocate(APi, AP_i, PRM)

      stop
      end program particle_filter




