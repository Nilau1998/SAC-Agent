\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{iat}[2020/05/15 v1 Template]
\PassOptionsToClass{%
	a4paper,%
	11pt,%
	parskip=half,%
	%twoside,%
	%openright,%
	BCOR=3mm,%
}{scrreprt}
\ProcessOptions\relax
\LoadClass{scrreprt}



\PassOptionsToPackage{right=3.5cm,left=3.5cm,top=3.5cm,bottom=4cm,head=25pt,foot=21pt, footskip=1.1cm}{geometry}
\PassOptionsToPackage{hidelinks,breaklinks,linktocpage}{hyperref}
\PassOptionsToPackage{libertine}{newtxmath}
\PassOptionsToPackage{headsepline=3pt, footsepline, automark, autooneside=false}{scrlayer-scrpage}
\PassOptionsToPackage{english,main=ngerman}{babel}
\PassOptionsToPackage{detect-all}{siunitx}
\PassOptionsToPackage{style=ieee, urldate=comp, bibencoding=UTF8, backend=biber}{biblatex}
\PassOptionsToPackage{T1}{fontenc}
\PassOptionsToPackage{utf8}{inputenc}

\RequirePackage{geometry}

\RequirePackage{fontenc}
\RequirePackage{inputenc}

\RequirePackage{babel}
\RequirePackage{csquotes}
\RequirePackage{microtype}
\RequirePackage{setspace}% für Onehalfspacing, ggf irgendwie erstetzen

\RequirePackage{libertine}
\RequirePackage{newtxmath}
\RequirePackage{amsmath,amsfonts,amstext}

\RequirePackage{hyperref}
\RequirePackage{color}
\RequirePackage{graphicx}
\RequirePackage{siunitx}
\RequirePackage{textcomp}
\RequirePackage{biblatex}
\RequirePackage{glossaries}
\RequirePackage{scrlayer-scrpage}

% %---------------------------------------


%%Pagestyle--IAT------------
\definecolor{iatred}{rgb}{0.79,0,0}
\addtokomafont{headsepline}{\color{iatred}}

\newpairofpagestyles{iat}{
	\ihead{\headmark}
	\chead{}
	\ohead{%
		\begin{minipage}{0.1\linewidth}%
			\includegraphics[width=\linewidth]{Logos/iat_logo.pdf}%
		\end{minipage}
	}
	\lofoot{%
		\begin{minipage}{0.1\linewidth}%
			\vspace{1mm}
			\includegraphics[scale=1.2]{Logos/uni_logo.pdf}\relax%
		\end{minipage}%
	}
	\rofoot{\pagemark}
}

%Format Options
%-----------------------
\onehalfspacing
\renewcommand{\thesubsubsection}{\thesubsection.\alph{subsubsection}}
%Caption Formatting
\addtokomafont{captionlabel}{\sffamily\bfseries}
\setcapindent{0em}
\setcapwidth{1\textwidth}
\KOMAoptions{captions=nooneline}

%Figures Centered in Figure-Environment
\makeatletter
\g@addto@macro\@floatboxreset\centering
\makeatother

%\input{ctII_tutorials_glossaries.tex}
%\makeglossaries
\graphicspath{{./ctII_graphics/}}
\renewcommand*\chapterpagestyle{iat}



