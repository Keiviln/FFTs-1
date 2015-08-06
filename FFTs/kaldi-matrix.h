// matrix/kaldi‐matrix.h
// Copyright 2009‐2011 Ondrej Glembek; Microsoft Corporation; Lukas Burget;
// Saarland University; Petr Schwarz; Yanmin Qian;
// Karel Vesely; Go Vivace Inc.; Haihua Xu

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE‐2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON‐INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_MATRIX_KALDI_MATRIX_H_
#define KALDI_MATRIX_KALDI_MATRIX_H_ 1

#include "matrix/matrix‐common.h"

namespace kaldi {


	template<typename Real>
	Real TraceMatMat(const MatrixBase<Real> &A, const MatrixBase<Real> &B,
		MatrixTransposeType trans = kNoTrans);


	template<typename Real>
	class MatrixBase {
	public:
		// so this child can access protected members of other instances.
		friend class Matrix<Real>;
		// friend declarations for CUDA matrices (see ../cudamatrix/)
		friend class CuMatrixBase<Real>;
		friend class CuMatrix<Real>;
		friend class CuSubMatrix<Real>;
		friend class CuPackedMatrix<Real>;

		friend class PackedMatrix<Real>;

		inline MatrixIndexT NumRows() const { return num_rows_; }

		inline MatrixIndexT NumCols() const { return num_cols_; }

		inline MatrixIndexT Stride() const { return stride_; }

		size_t SizeInBytes() const {
			return static_cast<size_t>(num_rows_)* static_cast<size_t>(stride_)*
				sizeof(Real);
		}

		inline const Real* Data() const {
			return data_;
		}

		inline Real* Data() { return data_; }

		inline Real* RowData(MatrixIndexT i) {
			KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
				static_cast<UnsignedMatrixIndexT>(num_rows_));
			return data_ + i * stride_;
		}

		inline const Real* RowData(MatrixIndexT i) const {
			KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
				static_cast<UnsignedMatrixIndexT>(num_rows_));
			return data_ + i * stride_;
		}

		inline Real& operator() (MatrixIndexT r, MatrixIndexT c) {
			KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
				static_cast<UnsignedMatrixIndexT>(num_rows_) &&
				static_cast<UnsignedMatrixIndexT>(c) <
				static_cast<UnsignedMatrixIndexT>(num_cols_));
			return *(data_ + r * stride_ + c);
		}
		Real &Index(MatrixIndexT r, MatrixIndexT c) { return (*this)(r, c); }

		inline const Real operator() (MatrixIndexT r, MatrixIndexT c) const {
			KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
				static_cast<UnsignedMatrixIndexT>(num_rows_) &&
				static_cast<UnsignedMatrixIndexT>(c) <
				static_cast<UnsignedMatrixIndexT>(num_cols_));
			return *(data_ + r * stride_ + c);
		}

		/* Basic setting‐to‐special values functions. */

		void SetZero();
		void Set(Real);
		void SetUnit();
		void SetRandn();
		void SetRandUniform();

		/* Copying functions. These do not resize the matrix! */


		template<typename OtherReal>
		void CopyFromMat(const MatrixBase<OtherReal> & M,
			MatrixTransposeType trans = kNoTrans);

		void CopyFromMat(const CompressedMatrix &M);

		template<typename OtherReal>
		void CopyFromSp(const SpMatrix<OtherReal> &M);

		template<typename OtherReal>
		void CopyFromTp(const TpMatrix<OtherReal> &M,
			MatrixTransposeType trans = kNoTrans);

		template<typename OtherReal>
		void CopyFromMat(const CuMatrixBase<OtherReal> &M,
			MatrixTransposeType trans = kNoTrans);

		void CopyRowsFromVec(const VectorBase<Real> &v);

		void CopyRowsFromVec(const CuVectorBase<Real> &v);

		template<typename OtherReal>
		void CopyRowsFromVec(const VectorBase<OtherReal> &v);

		void CopyColsFromVec(const VectorBase<Real> &v);

		void CopyColFromVec(const VectorBase<Real> &v, const MatrixIndexT col);
		void CopyRowFromVec(const VectorBase<Real> &v, const MatrixIndexT row);
		void CopyDiagFromVec(const VectorBase<Real> &v);

		/* Accessing of sub‐parts of the matrix. */

		inline const SubVector<Real> Row(MatrixIndexT i) const {
			KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
				static_cast<UnsignedMatrixIndexT>(num_rows_));
			return SubVector<Real>(data_ + (i * stride_), NumCols());
		}

		inline SubVector<Real> Row(MatrixIndexT i) {
			KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
				static_cast<UnsignedMatrixIndexT>(num_rows_));
			return SubVector<Real>(data_ + (i * stride_), NumCols());
		}

		inline SubMatrix<Real> Range(const MatrixIndexT row_offset,
			const MatrixIndexT num_rows,
			const MatrixIndexT col_offset,
			const MatrixIndexT num_cols) const {
			return SubMatrix<Real>(*this, row_offset, num_rows,
				col_offset, num_cols);
		}
		inline SubMatrix<Real> RowRange(const MatrixIndexT row_offset,
			const MatrixIndexT num_rows) const {
			return SubMatrix<Real>(*this, row_offset, num_rows, 0, num_cols_);
		}
		inline SubMatrix<Real> ColRange(const MatrixIndexT col_offset,
			const MatrixIndexT num_cols) const {
			return SubMatrix<Real>(*this, 0, num_rows_, col_offset, num_cols);
		}

		/* Various special functions. */
		Real Sum() const;
		Real Trace(bool check_square = true) const;
		// If check_square = true, will crash if matrix is not square.

		Real Max() const;
		Real Min() const;

		void MulElements(const MatrixBase<Real> &A);

		void DivElements(const MatrixBase<Real> &A);

		void Scale(Real alpha);

		void Max(const MatrixBase<Real> &A);

		void MulColsVec(const VectorBase<Real> &scale);

		void MulRowsVec(const VectorBase<Real> &scale);

		void MulRowsGroupMat(const MatrixBase<Real> &src);

		Real LogDet(Real *det_sign = NULL) const;

		void Invert(Real *log_det = NULL, Real *det_sign = NULL,
			bool inverse_needed = true);
		void InvertDouble(Real *LogDet = NULL, Real *det_sign = NULL,
			bool inverse_needed = true);

		void InvertElements();

		void Transpose();

		void CopyCols(const MatrixBase<Real> &src,
			const std::vector<MatrixIndexT> &indices);

		void CopyRows(const MatrixBase<Real> &src,
			const std::vector<MatrixIndexT> &indices);

		void ApplyFloor(Real floor_val);

		void ApplyCeiling(Real ceiling_val);

		void ApplyLog();

		void ApplyExp();

		void ApplyPow(Real power);

		void ApplyPowAbs(Real power, bool include_sign = false);

		void ApplyHeaviside();

		void Eig(MatrixBase<Real> *P,
			VectorBase<Real> *eigs_real,
			VectorBase<Real> *eigs_imag) const;

		bool Power(Real pow);

		void DestructiveSvd(VectorBase<Real> *s, MatrixBase<Real> *U,
			MatrixBase<Real> *Vt); // Destroys calling matrix.

		void Svd(VectorBase<Real> *s, MatrixBase<Real> *U,
			MatrixBase<Real> *Vt) const;
		void Svd(VectorBase<Real> *s) const { Svd(s, NULL, NULL); }


		Real MinSingularValue() const {
			Vector<Real> tmp(std::min(NumRows(), NumCols()));
			Svd(&tmp);
			return tmp.Min();
		}

		void TestUninitialized() const; // This function is designed so that if any
		ment
			// if the matrix is uninitialized memory, valgrind will complain.

			Real Cond() const;

		bool IsSymmetric(Real cutoff = 1.0e‐05) const; // replace magic number

		bool IsDiagonal(Real cutoff = 1.0e‐05) const; // replace magic number

		bool IsUnit(Real cutoff = 1.0e‐05) const; // replace magic number

		bool IsZero(Real cutoff = 1.0e‐05) const; // replace magic number

		Real FrobeniusNorm() const;

		bool ApproxEqual(const MatrixBase<Real> &other, float tol = 0.01) const;

		bool Equal(const MatrixBase<Real> &other) const;

		Real LargestAbsElem() const; // largest absolute value.

		Real LogSumExp(Real prune = ‐1.0) const;

		Real ApplySoftMax();

		void Sigmoid(const MatrixBase<Real> &src);

		void SoftHinge(const MatrixBase<Real> &src);

		void GroupPnorm(const MatrixBase<Real> &src, Real power);


		void GroupPnormDeriv(const MatrixBase<Real> &input, const MatrixBase<Real>
			tput,
			Real power);


		void Tanh(const MatrixBase<Real> &src);

		// Function used in backpropagating derivatives of the sigmoid function:
		// element‐by‐element, set *this = diff * value * (1.0 ‐ value).
		void DiffSigmoid(const MatrixBase<Real> &value,
			const MatrixBase<Real> &diff);

		// Function used in backpropagating derivatives of the tanh function:
		// element‐by‐element, set *this = diff * (1.0 ‐ value^2).
		void DiffTanh(const MatrixBase<Real> &value,
			const MatrixBase<Real> &diff);

		void SymPosSemiDefEig(VectorBase<Real> *s, MatrixBase<Real> *P,
			Real check_thresh = 0.001);

		friend Real kaldi::TraceMatMat<Real>(const MatrixBase<Real> &A,
			const MatrixBase<Real> &B, MatrixTransposeType trans); // tr (A B)

		// so it can get around const restrictions on the pointer to data_.
		friend class SubMatrix<Real>;

		void Add(const Real alpha);

		void AddToDiag(const Real alpha);

		template<typename OtherReal>
		void AddVecVec(const Real alpha, const VectorBase<OtherReal> &a,
			const VectorBase<OtherReal> &b);

		template<typename OtherReal>
		void AddVecToRows(const Real alpha, const VectorBase<OtherReal> &v);

		template<typename OtherReal>
		void AddVecToCols(const Real alpha, const VectorBase<OtherReal> &v);

		void AddMat(const Real alpha, const MatrixBase<Real> &M,
			MatrixTransposeType transA = kNoTrans);

		void SymAddMat2(const Real alpha, const MatrixBase<Real> &M,
			MatrixTransposeType transA, Real beta);

		void AddDiagVecMat(const Real alpha, VectorBase<Real> &v,
			const MatrixBase<Real> &M, MatrixTransposeType transM,
			Real beta = 1.0);

		void AddMatDiagVec(const Real alpha,
			const MatrixBase<Real> &M, MatrixTransposeType transM,
			VectorBase<Real> &v,
			Real beta = 1.0);

		void AddMatMatElements(const Real alpha,
			const MatrixBase<Real>& A,
			const MatrixBase<Real>& B,
			const Real beta);

		template<typename OtherReal>
		void AddSp(const Real alpha, const SpMatrix<OtherReal> &S);

		void AddMatMat(const Real alpha,
			const MatrixBase<Real>& A, MatrixTransposeType transA,
			const MatrixBase<Real>& B, MatrixTransposeType transB,
			const Real beta);

		void AddMatMatDivMat(const MatrixBase<Real>& A,
			const MatrixBase<Real>& B,
			const MatrixBase<Real>& C);

		void AddMatSmat(const Real alpha,
			const MatrixBase<Real>& A, MatrixTransposeType transA,
			const MatrixBase<Real>& B, MatrixTransposeType transB,
			const Real beta);

		void AddSmatMat(const Real alpha,
			const MatrixBase<Real>& A, MatrixTransposeType transA,
			const MatrixBase<Real>& B, MatrixTransposeType transB,
			const Real beta);

		void AddMatMatMat(const Real alpha,
			const MatrixBase<Real>& A, MatrixTransposeType transA,
			const MatrixBase<Real>& B, MatrixTransposeType transB,
			const MatrixBase<Real>& C, MatrixTransposeType transC,
			const Real beta);

		// This and the routines below are really
		// stubs that need to be made more efficient.
		void AddSpMat(const Real alpha,
			const SpMatrix<Real>& A,
			const MatrixBase<Real>& B, MatrixTransposeType transB,
			const Real beta) {
			Matrix<Real> M(A);
			return AddMatMat(alpha, M, kNoTrans, B, transB, beta);
		}
		void AddTpMat(const Real alpha,
			const TpMatrix<Real>& A, MatrixTransposeType transA,
			const MatrixBase<Real>& B, MatrixTransposeType transB,
			const Real beta) {
			Matrix<Real> M(A);
			return AddMatMat(alpha, M, transA, B, transB, beta);
		}
		void AddMatSp(const Real alpha,
			const MatrixBase<Real>& A, MatrixTransposeType transA,
			const SpMatrix<Real>& B,
			const Real beta) {
			Matrix<Real> M(B);
			return AddMatMat(alpha, A, transA, M, kNoTrans, beta);
		}
		void AddSpMatSp(const Real alpha,
			const SpMatrix<Real> &A,
			const MatrixBase<Real>& B, MatrixTransposeType transB,
			const SpMatrix<Real>& C,
			const Real beta) {
			Matrix<Real> M(A), N(C);
			return AddMatMatMat(alpha, M, kNoTrans, B, transB, N, kNoTrans, beta);
		}
		void AddMatTp(const Real alpha,
			const MatrixBase<Real>& A, MatrixTransposeType transA,
			const TpMatrix<Real>& B, MatrixTransposeType transB,
			const Real beta) {
			Matrix<Real> M(B);
			return AddMatMat(alpha, A, transA, M, transB, beta);
		}

		void AddTpTp(const Real alpha,
			const TpMatrix<Real>& A, MatrixTransposeType transA,
			const TpMatrix<Real>& B, MatrixTransposeType transB,
			const Real beta) {
			Matrix<Real> M(A), N(B);
			return AddMatMat(alpha, M, transA, N, transB, beta);
		}

		// This one is more efficient, not like the others above.
		void AddSpSp(const Real alpha,
			const SpMatrix<Real>& A, const SpMatrix<Real>& B,
			const Real beta);

		void CopyLowerToUpper();

		void CopyUpperToLower();

		void OrthogonalizeRows();

		// Will throw exception on failure.
		void Read(std::istream & in, bool binary, bool add = false);
		void Write(std::ostream & out, bool binary) const;

		// Below is internal methods for Svd, user does not have to know about this.
#if !defined(HAVE_ATLAS) && !defined(USE_KALDI_SVD)
		// protected:
		// Should be protected but used directly in testing routine.
		// destroys *this!
		void LapackGesvd(VectorBase<Real> *s, MatrixBase<Real> *U,
			MatrixBase<Real> *Vt);
#else
	protected:
		// destroys *this!
		bool JamaSvd(VectorBase<Real> *s, MatrixBase<Real> *U,
			MatrixBase<Real> *V);

#endif
	protected:

		explicit MatrixBase(Real *data, MatrixIndexT cols, MatrixIndexT rows,
			rixIndexT stride) :
			data_(data), num_cols_(cols), num_rows_(rows), stride_(stride) {
			KALDI_ASSERT_IS_FLOATING_TYPE(Real);
		}

		explicit MatrixBase() : data_(NULL) {
			KALDI_ASSERT_IS_FLOATING_TYPE(Real);
		}

		// Make sure pointers to MatrixBase cannot be deleted.
		~MatrixBase() { }

		inline Real* Data_workaround() const {
			return data_;
		}

		Real* data_;

		MatrixIndexT num_cols_;
		MatrixIndexT num_rows_;

		MatrixIndexT stride_;
	private:
		KALDI_DISALLOW_COPY_AND_ASSIGN(MatrixBase);
	};

	template<typename Real>
	class Matrix : public MatrixBase<Real> {
	public:

		Matrix();

		Matrix(const MatrixIndexT r, const MatrixIndexT c,
			MatrixResizeType resize_type = kSetZero) :
			MatrixBase<Real>() { Resize(r, c, resize_type); }

		template<typename OtherReal>
		explicit Matrix(const CuMatrixBase<OtherReal> &cu,
			MatrixTransposeType trans = kNoTrans);


		void Swap(Matrix<Real> *other);

		void Swap(CuMatrix<Real> *mat);

		explicit Matrix(const MatrixBase<Real> & M,
			MatrixTransposeType trans = kNoTrans);

		Matrix(const Matrix<Real> & M); // (cannot make explicit)

		template<typename OtherReal>
		explicit Matrix(const MatrixBase<OtherReal> & M,
			MatrixTransposeType trans = kNoTrans);

		template<typename OtherReal>
		explicit Matrix(const SpMatrix<OtherReal> & M) : MatrixBase<Real>() {
			Resize(M.NumRows(), M.NumRows(), kUndefined);
			this‐>CopyFromSp(M);
		}

		explicit Matrix(const CompressedMatrix &C);

		template <typename OtherReal>
		explicit Matrix(const TpMatrix<OtherReal> & M,
			MatrixTransposeType trans = kNoTrans) : MatrixBase<Real>() {
			if (trans == kNoTrans) {
				Resize(M.NumRows(), M.NumCols(), kUndefined);
				this‐>CopyFromTp(M);
			}
			else {
				Resize(M.NumCols(), M.NumRows(), kUndefined);
				this‐>CopyFromTp(M, kTrans);
			}
		}

		// Unlike one in base, allows resizing.
		void Read(std::istream & in, bool binary, bool add = false);

		void RemoveRow(MatrixIndexT i);

		void Transpose();

		~Matrix() { Destroy(); }

		void Resize(const MatrixIndexT r,
			const MatrixIndexT c,
			MatrixResizeType resize_type = kSetZero);

		Matrix<Real> &operator = (const MatrixBase<Real> &other) {
			if (MatrixBase<Real>::NumRows() != other.NumRows() ||
				MatrixBase<Real>::NumCols() != other.NumCols())
				Resize(other.NumRows(), other.NumCols(), kUndefined);
			MatrixBase<Real>::CopyFromMat(other);
			return *this;
		}

		Matrix<Real> &operator = (const Matrix<Real> &other) {
			if (MatrixBase<Real>::NumRows() != other.NumRows() ||
				MatrixBase<Real>::NumCols() != other.NumCols())
				Resize(other.NumRows(), other.NumCols(), kUndefined);
			MatrixBase<Real>::CopyFromMat(other);
			return *this;
		}


	private:
		void Destroy();

		void Init(const MatrixIndexT r,
			const MatrixIndexT c);

	};


	struct HtkHeader {
		int32 mNSamples;
		int32 mSamplePeriod;
		int16 mSampleSize;
		uint16 mSampleKind;
	};

	// Read HTK formatted features from file into matrix.
	template<typename Real>
	bool ReadHtk(std::istream &is, Matrix<Real> *M, HtkHeader *header_ptr);

	// Write (HTK format) features to file from matrix.
	template<typename Real>
	bool WriteHtk(std::ostream &os, const MatrixBase<Real> &M, HtkHeader htk_hdr);

	// Write (CMUSphinx format) features to file from matrix.
	template<typename Real>
	bool WriteSphinx(std::ostream &os, const MatrixBase<Real> &M);


	template<typename Real>
	class SubMatrix : public MatrixBase<Real> {
	public:
		// Initialize a SubMatrix from part of a matrix; this is
		// a bit like A(b:c, d:e) in Matlab.
		// This initializer is against the proper semantics of "const", since
		// SubMatrix can change its contents. It would be hard to implement
		// a "const‐safe" version of this class.
		SubMatrix(const MatrixBase<Real>& T,
			const MatrixIndexT ro, // row offset, 0 < ro < NumRows()
			const MatrixIndexT r, // number of rows, r > 0
			const MatrixIndexT co, // column offset, 0 < co < NumCols()
			const MatrixIndexT c); // number of columns, c > 0

		// This initializer is mostly intended for use in CuMatrix and related
		// classes. Be careful!
		SubMatrix(Real *data,
			MatrixIndexT num_rows,
			MatrixIndexT num_cols,
			MatrixIndexT stride);

		~SubMatrix<Real>() {}

		SubMatrix<Real>(const SubMatrix &other) :
			MatrixBase<Real>(other.data_, other.num_cols_, other.num_rows_,
			other.stride_) {}

	private:
		SubMatrix<Real> &operator = (const SubMatrix<Real> &other);
	};


	// Some declarations. These are traces of products.


	template<typename Real>
	bool ApproxEqual(const MatrixBase<Real> &A,
		const MatrixBase<Real> &B, Real tol = 0.01) {
		return A.ApproxEqual(B, tol);
	}

	template<typename Real>
	inline void AssertEqual(const MatrixBase<Real> &A, const MatrixBase<Real> &B,
		float tol = 0.01) {
		KALDI_ASSERT(A.ApproxEqual(B, tol));
	}

	template <typename Real>
	double TraceMat(const MatrixBase<Real> &A) { return A.Trace(); }


	template <typename Real>
	Real TraceMatMatMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
		const MatrixBase<Real> &B, MatrixTransposeType transB,
		const MatrixBase<Real> &C, MatrixTransposeType transC);

	template <typename Real>
	Real TraceMatMatMatMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
		const MatrixBase<Real> &B, MatrixTransposeType transB,
		const MatrixBase<Real> &C, MatrixTransposeType transC,
		const MatrixBase<Real> &D, MatrixTransposeType transD);





	template<typename Real> void SortSvd(VectorBase<Real> *s, MatrixBase<Real> *U,
		MatrixBase<Real>* Vt = NULL,
		bool sort_on_absolute_value = true);

	template<typename Real>
	void CreateEigenvalueMatrix(const VectorBase<Real> &real, const VectorBase<Real>
		ag,
		MatrixBase<Real> *D);

	template<typename Real>
	bool AttemptComplexPower(Real *x_re, Real *x_im, Real power);




	template<typename Real>
	std::ostream & operator << (std::ostream & Out, const MatrixBase<Real> & M);

	template<typename Real>
	std::istream & operator >> (std::istream & In, MatrixBase<Real> & M);

	// The Matrix read allows resizing, so we override the MatrixBase one.
	template<typename Real>
	std::istream & operator >> (std::istream & In, Matrix<Real> & M);


	template<typename Real>
	bool SameDim(const MatrixBase<Real> &M, const MatrixBase<Real> &N) {
		return (M.NumRows() == N.NumRows() && M.NumCols() == N.NumCols());
	}



} // namespace kaldi



// we need to include the implementation and some
// template specializations.
#include "kaldi‐matrix‐inl.h"


#endif // KALDI_MATRIX_KALDI_MATRIX_H_