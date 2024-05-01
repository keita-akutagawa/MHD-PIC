# MHD-PIC

連結階層シミュレーションのコードです。 \
IdealMHDとPICを接続領域を用いて繋げます。 \
1次元は、MHD→PIC・PIC→MHDのfast modeの伝搬に成功しました。 \

現在はnumpyを駆使して書いていますが、いずれC++に移行します。
さらにThrustライブラリ等を用いてGPU対応させます。 \
※今のコードもnumpyをcupyにすればGPU対応可能です。


【参考文献】
- T. Sugiyama & K. Kusano, J. Comput. Phys., 227, 1340, 2007 
- S. Usami et al., Phys. Plasmas., 20, 061208, 2013 
- Lars K.S. Daldorff et al., J. Comput. Phys. 268, 236, 2014
- K.D. Makwana et al., Comput. Phys. Comm. 221, 81, 2017
