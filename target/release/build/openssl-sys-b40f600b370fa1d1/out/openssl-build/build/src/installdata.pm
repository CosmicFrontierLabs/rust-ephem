package OpenSSL::safe::installdata;

use strict;
use warnings;
use Exporter;
our @ISA = qw(Exporter);
our @EXPORT = qw(
    @PREFIX
    @libdir
    @BINDIR @BINDIR_REL_PREFIX
    @LIBDIR @LIBDIR_REL_PREFIX
    @INCLUDEDIR @INCLUDEDIR_REL_PREFIX
    @APPLINKDIR @APPLINKDIR_REL_PREFIX
    @ENGINESDIR @ENGINESDIR_REL_LIBDIR
    @MODULESDIR @MODULESDIR_REL_LIBDIR
    @PKGCONFIGDIR @PKGCONFIGDIR_REL_LIBDIR
    @CMAKECONFIGDIR @CMAKECONFIGDIR_REL_LIBDIR
    $VERSION @LDLIBS
);

our @PREFIX                     = ( '/home/runner/work/rust-ephem/rust-ephem/target/release/build/openssl-sys-b40f600b370fa1d1/out/openssl-build/install' );
our @libdir                     = ( '/home/runner/work/rust-ephem/rust-ephem/target/release/build/openssl-sys-b40f600b370fa1d1/out/openssl-build/install/lib' );
our @BINDIR                     = ( '/home/runner/work/rust-ephem/rust-ephem/target/release/build/openssl-sys-b40f600b370fa1d1/out/openssl-build/install/bin' );
our @BINDIR_REL_PREFIX          = ( 'bin' );
our @LIBDIR                     = ( '/home/runner/work/rust-ephem/rust-ephem/target/release/build/openssl-sys-b40f600b370fa1d1/out/openssl-build/install/lib' );
our @LIBDIR_REL_PREFIX          = ( 'lib' );
our @INCLUDEDIR                 = ( '/home/runner/work/rust-ephem/rust-ephem/target/release/build/openssl-sys-b40f600b370fa1d1/out/openssl-build/install/include' );
our @INCLUDEDIR_REL_PREFIX      = ( 'include' );
our @APPLINKDIR                 = ( '/home/runner/work/rust-ephem/rust-ephem/target/release/build/openssl-sys-b40f600b370fa1d1/out/openssl-build/install/include/openssl' );
our @APPLINKDIR_REL_PREFIX      = ( 'include/openssl' );
our @ENGINESDIR                 = ( '/home/runner/work/rust-ephem/rust-ephem/target/release/build/openssl-sys-b40f600b370fa1d1/out/openssl-build/install/lib/engines-3' );
our @ENGINESDIR_REL_LIBDIR      = ( 'engines-3' );
our @MODULESDIR                 = ( '/home/runner/work/rust-ephem/rust-ephem/target/release/build/openssl-sys-b40f600b370fa1d1/out/openssl-build/install/lib/ossl-modules' );
our @MODULESDIR_REL_LIBDIR      = ( 'ossl-modules' );
our @PKGCONFIGDIR               = ( '/home/runner/work/rust-ephem/rust-ephem/target/release/build/openssl-sys-b40f600b370fa1d1/out/openssl-build/install/lib/pkgconfig' );
our @PKGCONFIGDIR_REL_LIBDIR    = ( 'pkgconfig' );
our @CMAKECONFIGDIR             = ( '/home/runner/work/rust-ephem/rust-ephem/target/release/build/openssl-sys-b40f600b370fa1d1/out/openssl-build/install/lib/cmake/OpenSSL' );
our @CMAKECONFIGDIR_REL_LIBDIR  = ( 'cmake/OpenSSL' );
our $VERSION                    = '3.5.4';
our @LDLIBS                     =
    # Unix and Windows use space separation, VMS uses comma separation
    $^O eq 'VMS'
    ? split(/ *, */, '-ldl -pthread ')
    : split(/ +/, '-ldl -pthread ');

1;
